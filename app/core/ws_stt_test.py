import asyncio, json, time
import numpy as np
import soundfile as sf
import websockets

WS_URL = "ws://127.0.0.1:8000/ws/stream"
WAV_PATH = "input1.wav"   # 너 녹음 파일 경로
CHUNK_MS = 50             # 20~100ms 추천

def to_pcm16_mono_16k(wav_path: str):
    audio, sr = sf.read(wav_path, dtype="float32")
    # stereo -> mono
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    # resample 필요하면 여기서 처리 (지금은 sr==16000 가정)
    if sr != 16000:
        raise ValueError(f"sample rate must be 16000 for this test, got {sr}")
    # float32 [-1,1] -> int16
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    return pcm16.tobytes(), sr

async def main():
    pcm_bytes, sr = to_pcm16_mono_16k(WAV_PATH)

    bytes_per_ms = int(sr * 2 / 1000)  # 16kHz * 2bytes(int16) / 1000ms
    chunk_size = bytes_per_ms * CHUNK_MS

    async with websockets.connect(WS_URL, max_size=10_000_000) as ws:
        # 1) start
        await ws.send(json.dumps({
            "type": "start",
            "language_code": "ko-KR",
            "sample_rate_hz": 16000,
            "interim_results": True
        }))

        async def receiver():
            async for msg in ws:
                data = json.loads(msg)
                t = data.get("type")
                if t in ("stt_interim", "stt_final"):
                    print(f"[{t}] {data.get('text')}")
                else:
                    print("[recv]", data)

        recv_task = asyncio.create_task(receiver())

        # 2) 파일을 실시간처럼 청크로 전송
        for i in range(0, len(pcm_bytes), chunk_size):
            await ws.send(pcm_bytes[i:i+chunk_size])  # binary frame
            await asyncio.sleep(CHUNK_MS / 1000.0)

        # 3) stop
        await ws.send(json.dumps({"type": "stop"}))

        # 조금 기다렸다 종료
        await asyncio.sleep(1.0)
        recv_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())