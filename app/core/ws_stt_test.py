import asyncio
import json
import wave
import numpy as np
import soundfile as sf
import websockets
import base64
import time


#WS_URL = "ws://localhost:8082/ws/v1/kiosk/stt?sessionId=b5e95ed7-aac2-483b-aa03-71b3b66fa5d6&siteId=5&languageCode=ko-KR&token=kiosk-south-01-test"
# 스크립트 위치 기준 상대 경로로 고정 (실행 디렉터리와 무관)
WS_URL = "ws://localhost:8000/ws/stream"
WAV_PATH = r"C:\Users\문현우\Project_Guideon\guideon-AI\중국어3.wav"

# Cartesia 테스트용 voice_id (None이면 서버 환경변수 기본값 사용)
CARTESIA_VOICE_ID = "6fd8e3fc-a59d-479f-a072-b4f7e8284a78"

CHUNK_MS = 400
SAVE_TTS_AUDIO = True

# Cartesia pcm_s16le 출력 규격
PCM_SAMPLE_RATE = 24000
PCM_CHANNELS = 1
PCM_SAMPWIDTH = 2  # 16-bit = 2 bytes

START_PAYLOAD = {
    "type": "start",
    "siteId": 2,
    "language_code": "auto",  # 클라이언트 언어 감지 → 서버에서 실제 감지된 언어로 profile 갱신
    "ttsVoiceId": CARTESIA_VOICE_ID,  # None이면 서버 기본 voice 사용
}


def to_pcm16_mono_16k(wav_path: str):
    audio, sr = sf.read(wav_path, dtype="float32")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != 16000:
        # scipy 없이 numpy 선형 보간으로 리샘플링 (테스트 스크립트 용도로 충분)
        n = int(len(audio) * 16000 / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)
        sr = 16000

    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)

    return pcm16.tobytes(), sr


async def main():

    pcm_bytes, sr = to_pcm16_mono_16k(WAV_PATH)

    bytes_per_ms = int(sr * 2 / 1000)
    chunk_size = bytes_per_ms * CHUNK_MS

    async with websockets.connect(WS_URL, max_size=20_000_000) as ws:

        t_stt_final = None
        t_llm_first = None
        t_tts_first = None

        tts_audio = []       # (bytes, audio_format) 리스트
        tts_format = None    # 첫 청크에서 결정 ("pcm_s16le" or "mp3")

        async def receiver():
            nonlocal t_stt_final, t_llm_first, t_tts_first, tts_format

            async for msg in ws:

                data = json.loads(msg)
                t = data.get("type")

                if t in ("stt_interim", "stt_final"):

                    lang = data.get("language_code", "?")
                    print(f"[{t}] lang={lang} | {data.get('text')}")

                    if t == "stt_final" and t_stt_final is None:
                        t_stt_final = time.perf_counter()

                elif t == "llm_sentence":

                    lang = data.get("language_code", "?")
                    print(f"[LLM] lang={lang} | {data.get('text')}")

                    if t_llm_first is None:
                        t_llm_first = time.perf_counter()

                elif t == "tts_chunk":

                    lang = data.get("language_code", "?")
                    fmt  = data.get("audio_format", "mp3")
                    engine = "Cartesia" if fmt == "pcm_s16le" else "Google"
                    print(f"[TTS/{engine}] lang={lang} fmt={fmt} | {data.get('text')}")

                    if t_tts_first is None:
                        t_tts_first = time.perf_counter()

                    if SAVE_TTS_AUDIO:
                        audio = base64.b64decode(data["audio_b64"])
                        if tts_format is None:
                            tts_format = fmt
                        elif fmt != tts_format:
                            raise RuntimeError(
                                f"TTS audio_format이 스트림 중간에 변경됨: {tts_format} → {fmt}"
                            )
                        tts_audio.append(audio)

                elif t == "final_text":

                    print("\n[FINAL ANSWER]")
                    print("lang    :", data.get("language_code", "?"))
                    print("answer  :", data.get("answer"))
                    print("category:", data.get("category"))

                elif t == "done":

                    print("\n[SERVER DONE]")
                    break

                else:
                    print("[recv]", data)

        await ws.send(json.dumps(START_PAYLOAD))
        t0 = time.perf_counter()

        recv_task = asyncio.create_task(receiver())

        for i in range(0, len(pcm_bytes), chunk_size):
            await ws.send(pcm_bytes[i:i + chunk_size])
            await asyncio.sleep(CHUNK_MS / 1000)

        await ws.send(json.dumps({"type": "stop"}))

        await recv_task

        print("\n===== LATENCY =====")

        if t_stt_final:
            print("STT latency:", round((t_stt_final - t0) * 1000), "ms")

        if t_llm_first and t_stt_final:
            print("LLM first token:", round((t_llm_first - t_stt_final) * 1000), "ms")

        if t_tts_first and t_llm_first:
            print("TTS first audio:", round((t_tts_first - t_llm_first) * 1000), "ms")

        if SAVE_TTS_AUDIO and tts_audio:

            if tts_format == "pcm_s16le":
                # Cartesia: raw PCM → WAV 헤더 붙여서 저장
                with wave.open("tts_result.wav", "wb") as wf:
                    wf.setnchannels(PCM_CHANNELS)
                    wf.setsampwidth(PCM_SAMPWIDTH)
                    wf.setframerate(PCM_SAMPLE_RATE)
                    for chunk in tts_audio:
                        wf.writeframes(chunk)
                print("saved TTS audio → tts_result.wav  (Cartesia pcm_s16le, 24000Hz)")
            else:
                # Google TTS: MP3 그대로 저장
                with open("tts_result.mp3", "wb") as f:
                    for chunk in tts_audio:
                        f.write(chunk)
                print("saved TTS audio → tts_result.mp3  (Google TTS mp3)")


if __name__ == "__main__":
    asyncio.run(main())