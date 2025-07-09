from fastapi import FastAPI, File, UploadFile
from faster_whisper import WhisperModel
import uvicorn
import os
import tempfile


os.add_dll_directory("c:\\Program Files\\NVIDIA\\CUDNN\\v9.8\\bin\\12.8")

app = FastAPI()

# Configure your Whisper model (adjust device and compute_type as needed) compute_type="float16"cuda
MODEL_SIZE = "small"
model = WhisperModel(MODEL_SIZE, device="auto" )

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary file
    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Transcribe the audio using faster_whisper
    segments, info = model.transcribe(tmp_path, beam_size=5)
    os.remove(tmp_path)  # Clean up temporary file

    # Prepare a JSON-friendly response
    result_segments = [{"start": seg.start, "end": seg.end, "text": seg.text} for seg in segments]
    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": result_segments
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
