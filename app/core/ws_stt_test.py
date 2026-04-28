import asyncio
import json
import numpy as np
import soundfile as sf
import websockets
import base64
import time

WS_URL = "ws://localhost:8082/ws/v1/kiosk/stt?sessionId=b5e95ed7-aac2-483b-aa03-71b3b66fa5d6&siteId=5&languageCode=ko-KR&token=kiosk-south-01-test"
WAV_PATH = "ko_04.wav"

CHUNK_MS = 50
SAVE_TTS_AUDIO = True


def to_pcm16_mono_16k(wav_path: str):
    audio, sr = sf.read(wav_path, dtype="float32")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != 16000:
        raise ValueError(f"sample rate must be 16000, got {sr}")

    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)

    return pcm16.tobytes(), sr


async def main():

    pcm_bytes, sr = to_pcm16_mono_16k(WAV_PATH)

    bytes_per_ms = int(sr * 2 / 1000)
    chunk_size = bytes_per_ms * CHUNK_MS

    async with websockets.connect(WS_URL, max_size=20_000_000) as ws:

        t0 = time.perf_counter()

        t_stt_final = None
        t_llm_first = None
        t_tts_first = None

        tts_audio = []

        async def receiver():
            nonlocal t_stt_final, t_llm_first, t_tts_first

            async for msg in ws:

                data = json.loads(msg)
                t = data.get("type")

                if t in ("stt_interim", "stt_final"):

                    print(f"[{t}] {data.get('text')}")

                    if t == "stt_final" and t_stt_final is None:
                        t_stt_final = time.perf_counter()

                elif t == "llm_sentence":

                    print("[LLM]", data.get("text"))

                    if t_llm_first is None:
                        t_llm_first = time.perf_counter()

                elif t == "tts_chunk":

                    print("[TTS]", data.get("text"))

                    if t_tts_first is None:
                        t_tts_first = time.perf_counter()

                    if SAVE_TTS_AUDIO:
                        audio = base64.b64decode(data["audio_b64"])
                        tts_audio.append(audio)

                elif t == "final_text":

                    print("\n[FINAL ANSWER]")
                    print("answer  :", data.get("answer"))
                    print("category:", data.get("category"))

                elif t == "done":

                    print("\n[SERVER DONE]")
                    break

                else:
                    print("[recv]", data)

        recv_task = asyncio.create_task(receiver())

        for i in range(0, len(pcm_bytes), chunk_size):
            await ws.send(pcm_bytes[i:i + chunk_size])
            await asyncio.sleep(CHUNK_MS / 1000)

        await ws.send(json.dumps({"type": "stop"}))

        await recv_task

        print("\n===== LATENCY =====")

        if t_stt_final:
            print("STT latency:", round((t_stt_final - t0) * 1000), "ms")

        if t_llm_first:
            print("LLM first token:", round((t_llm_first - t_stt_final) * 1000), "ms")

        if t_tts_first:
            print("TTS first audio:", round((t_tts_first - t_llm_first) * 1000), "ms")

        if SAVE_TTS_AUDIO and tts_audio:

            with open("tts_result.mp3", "wb") as f:
                for chunk in tts_audio:
                    f.write(chunk)

            print("saved TTS audio → tts_result.mp3")


if __name__ == "__main__":
    asyncio.run(main())