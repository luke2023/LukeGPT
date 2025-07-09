from melo.api import TTS
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import io
import wave
import asyncio
import uvicorn
import sys

# 針對 Windows 設定事件迴圈策略
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = FastAPI()

# 設置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 TTS 模型
speed = 1.16  # 1.16
device = 'cuda:0'  # or cuda:0 if you have GPU cuda:0
model = TTS(language="ZH", device="cuda:0")
speaker_ids = model.hps.data.spk2id

def numpy_to_wav_bytes(audio_data: np.ndarray, sample_rate: int) -> bytes:
    """將 numpy 數組轉換為 WAV 格式的字節數據，並提高音量"""
    buffer = io.BytesIO()
    gain = 1.7  # 增益係數，聲音變大
    amplified = np.clip(audio_data * gain, -1.0, 1.0)  # 限制範圍避免破音

    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # 單聲道
        wav_file.setsampwidth(2)   # 16 位
        wav_file.setframerate(sample_rate)
        audio_int16 = (amplified * 32767).astype(np.int16)
        wav_file.writeframes(audio_int16.tobytes())
    return buffer.getvalue()

@app.get("/tts")
async def text_to_speech(text: str):
    try:
        audio_data = model.tts_to_file(text, speaker_ids['ZH'], speed=speed)
        wav_bytes = numpy_to_wav_bytes(audio_data, sample_rate=model.hps.data.sampling_rate)
        return Response(wav_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def run_server():
    """異步啟動 Uvicorn，處理 OSError 並自動重啟"""
    while True:
        try:
            config = uvicorn.Config(app, host="0.0.0.0", port=1234, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()
        except OSError as e:
            print(f"[Warning] OSError occurred: {e}, restarting server...")
            await asyncio.sleep(2)  # 避免過快重啟造成資源佔用

def handle_exception(loop, context):
    """全局異常處理器，針對 OSError(64) 進行吞掉"""
    exception = context.get("exception")
    if isinstance(exception, OSError):
        # 針對錯誤號 64 進行處理，忽略這個錯誤
        if exception.errno == 64:
            print(f"Swallowed OSError: {exception}")
            return
    # 其他異常照常打印
    print("Unhandled exception:", context)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(handle_exception)
    loop.create_task(run_server())
    loop.run_forever()
