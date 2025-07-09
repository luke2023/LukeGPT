import time
import threading
import queue
import re
import io
import wave
import requests

import sounddevice as sd
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import simpleaudio as sa
import webrtcvad  # pip install webrtcvad

from langchain_community.llms import Ollama

# ======== Configuration ========

# Audio recording parameters for VAD recording
SAMPLE_RATE = 16000  # VAD expects 16 kHz mono 16-bit PCM
CHANNELS = 1
FRAME_DURATION_MS = 30   # Duration for each frame for VAD (10, 20, or 30 ms)
SILENCE_DURATION_MS = 1800  # Stop recording after 1 second of silence
RECORDING_FILE = "recording.wav"

# URL of the FastAPI STT server endpoint
FASTAPI_URL = "http://localhost:8000/transcribe"

# TTS server URL (replace with your actual TTS server URL)
TTS_SERVER_URL = "http://localhost:1234/tts"

# Initialize the Ollama model for TTS streaming
llm = Ollama(model="gemma3:1b")

# Queue for TTS audio playback
audio_queue = queue.Queue(maxsize=3)

# ======== Utility Functions ========

def record_until_silence(output_filename: str,
                          sample_rate: int = SAMPLE_RATE,
                          channels: int = CHANNELS,
                          frame_duration_ms: int = FRAME_DURATION_MS,
                          silence_duration_ms: int = SILENCE_DURATION_MS):
    """
    Record audio from the microphone and automatically stop when silence is detected.
    Uses webrtcvad for voice activity detection.
    """
    vad = webrtcvad.Vad(2)  # Aggressiveness mode (0-3)
    frame_size = int(sample_rate * frame_duration_ms / 1000)  # samples per frame
    bytes_per_frame = frame_size * 2  # 16-bit audio => 2 bytes per sample
    silence_frames = int(silence_duration_ms / frame_duration_ms)
    silence_counter = 0
    recorded_frames = []

    print("Start recording. Please speak...")
    with sd.RawInputStream(samplerate=sample_rate, blocksize=frame_size,
                             dtype='int16', channels=channels) as stream:
        while True:
            # Read a frame from the microphone (raw bytes)
            frame, overflowed = stream.read(frame_size)
            if overflowed:
                print("Audio buffer overflow!")
            # Check if the frame contains speech
            is_speech = vad.is_speech(frame, sample_rate)
            if not is_speech:
                silence_counter += 1
            else:
                silence_counter = 0
            recorded_frames.append(frame)
            # If silence has been detected for enough consecutive frames, stop recording.
            if silence_counter >= silence_frames:
                print("Silence detected. Stopping recording.")
                break

    # Write the recorded frames to a WAV file
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(recorded_frames))
    print(f"Recording saved to {output_filename}")

def get_transcription(file_path: str) -> str:
    """Send the recorded audio file to the FastAPI server and get the transcription."""
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f)}
        response = requests.post(FASTAPI_URL, files=files)
    if response.status_code == 200:
        data = response.json()
        segments = data.get("segments", [])
        # Combine all segment texts into a full transcription
        full_text = " ".join(segment["text"] for segment in segments)
        print("Transcription received from server:")
        print(full_text)
        return full_text
    else:
        print(f"Error from transcription server: {response.status_code} {response.text}")
        return ""

def split_text(text: str):
    """Split text into sentences based on punctuation."""
    sentences = re.split(r'([。！？；])', text)
    processed_sentences = []
    temp_sentence = ""
    for part in sentences:
        if part in "。！？；":
            temp_sentence += part
            processed_sentences.append(temp_sentence.strip())
            temp_sentence = ""
        else:
            temp_sentence += part
    return [s.strip() for s in processed_sentences if s.strip()]

def fetch_tts(sentence: str):
    """Fetch TTS audio from the server for a given sentence."""
    try:
        response = requests.get(
            TTS_SERVER_URL,
            params={"text": sentence},
            headers={"User-Agent": "TTS-Client/1.0"},
            stream=True
        )
        if response.status_code == 200:
            audio = AudioSegment.from_file(io.BytesIO(response.content), format="wav")
            return audio
        else:
            print(f"TTS error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"TTS request failed: {e}")
        return None

def audio_loader(sentences):
    """Background thread to fetch TTS audio for each sentence."""
    for sentence in sentences:
        print(f"Fetching TTS for sentence: {sentence}")
        audio = fetch_tts(sentence)
        if audio:
            audio_queue.put(audio)
        else:
            print(f"Skipping sentence due to TTS failure: {sentence}")

def audio_player():
    """Thread to play audio from the queue with increased volume."""
    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        
        # Increase volume by 6 dB (adjust as needed)
        amplified_audio = audio + 10

        raw_data = amplified_audio.raw_data
        sample_rate = amplified_audio.frame_rate
        num_channels = amplified_audio.channels
        bytes_per_sample = amplified_audio.sample_width

        play_obj = sa.play_buffer(raw_data, num_channels, bytes_per_sample, sample_rate)
        play_obj.wait_done()
        time.sleep(0)

def stream_tts(query: str):
    """Real-time TTS streaming using the Ollama model."""
    current_text = ""
    processed_sentences_set = set()
    loader_threads = []

    # Start the audio player thread
    player_thread = threading.Thread(target=audio_player)
    player_thread.start()

    new_sentences = []

    for chunk in llm.stream(query):
        # Remove asterisk symbols
        chunk = re.sub(r"[*]", "", chunk)
        current_text += chunk
        sentences = split_text(current_text)
        for sentence in sentences:
            if sentence not in processed_sentences_set:
                new_sentences.append(sentence)
                processed_sentences_set.add(sentence)
        if new_sentences:
            print(f"New sentences for TTS: {new_sentences}")
            loader_thread = threading.Thread(target=audio_loader, args=(new_sentences,))
            loader_thread.start()
            loader_threads.append(loader_thread)
            new_sentences = []

    # Wait for all loader threads to complete
    for thread in loader_threads:
        thread.join()

    # Signal the audio player to stop
    audio_queue.put(None)
    player_thread.join()

# ======== Main Flow ========

if __name__ == "__main__":
    while(True): 
        # Step 1: Record audio until silence is detected
        record_until_silence(RECORDING_FILE)

        # Optional: Convert recording to another format if needed (e.g. using soundfile)
        # Step 2: Get transcription by sending audio to the FastAPI server
        transcribed_text = get_transcription(RECORDING_FILE)
        if not transcribed_text:
            print("No transcription obtained. Exiting.")
            exit(1)

        # Step 3: Run TTS streaming using the transcribed text
        print("\nStarting TTS streaming with Ollama...")
        stream_tts(transcribed_text)
