"""
STT 배치 테스트
wav/ 폴더의 모든 WAV 파일을 WebSocket으로 전송하고
STT 결과와 latency를 results.csv에 저장한다.

사용법:
    cd guideon-AI
    python app/core/stt_test.py
    python app/core/stt_test.py --wav-dir ./wav --manifest ./manifest.csv --output ./results.csv
"""
import asyncio
import csv
import json
import logging
import sys
import time
import wave
from pathlib import Path
import argparse

try:
    import websockets
except ImportError:
    print("[ERROR] pip install websockets")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("stt_test")

WS_URL       = "ws://127.0.0.1:8000/ws/stream"
CHUNK_MS     = 50
SLEEP_SEC    = CHUNK_MS / 1000


def load_manifest(path: Path) -> dict[str, dict]:
    """manifest.csv → {file_name: {language_code, reference_text}}"""
    result = {}
    if not path.exists():
        return result
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["file_name"].strip()] = {
                "language_code":  row.get("language_code", "ko").strip(),
                "reference_text": row.get("reference_text", "").strip(),
            }
    return result


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def wav_to_pcm_chunks(path: Path) -> list[bytes]:
    with wave.open(str(path), "rb") as w:
        sr       = w.getframerate()
        pcm      = w.readframes(w.getnframes())
    chunk_size = (sr * 2 * CHUNK_MS) // 1000   # 16-bit mono
    return [pcm[i:i + chunk_size] for i in range(0, len(pcm), chunk_size)]


async def run_one(wav_path: Path, language_code: str) -> dict:
    """단일 WAV 파일 테스트. stt_final 텍스트와 latency 반환."""
    chunks   = wav_to_pcm_chunks(wav_path)
    duration = wav_duration(wav_path)

    t_start          = None
    t_first_interim  = None
    t_stt_final      = None
    stt_text         = ""
    done_event       = asyncio.Event()

    start_msg = json.dumps({
        "type":           "start",
        "site_id":        1,
        "language_code":  language_code,
        "sample_rate_hz": 16000,
        "interim_results": True,
        "tts_stream":     False,
    })

    async with websockets.connect(WS_URL) as ws:

        await ws.send(start_msg)

        async def sender():
            nonlocal t_start
            t_start = time.perf_counter()
            for chunk in chunks:
                await ws.send(chunk)
                await asyncio.sleep(SLEEP_SEC)
            await ws.send(json.dumps({"type": "stop"}))

        async def receiver():
            nonlocal t_first_interim, t_stt_final, stt_text
            try:
                async for raw in ws:
                    msg  = json.loads(raw)
                    t    = msg.get("type", "")
                    now  = time.perf_counter()

                    if t == "stt_interim" and t_first_interim is None:
                        t_first_interim = now

                    elif t == "stt_final":
                        stt_text    = msg.get("text", "")
                        t_stt_final = now

                    elif t in ("done", "error"):
                        break
            except Exception:
                pass
            finally:
                done_event.set()

        send_task = asyncio.create_task(sender())
        recv_task = asyncio.create_task(receiver())

        await send_task
        try:
            await asyncio.wait_for(done_event.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            log.warning("  timeout: %s", wav_path.name)

        recv_task.cancel()

    t_end = time.perf_counter()

    def elapsed(t): return round(t - t_start, 3) if t and t_start else -1.0

    return {
        "file_name":            wav_path.name,
        "language_code":        language_code,
        "predicted_text":       stt_text,
        "audio_duration":       round(duration, 3),
        "time_to_first_interim": elapsed(t_first_interim),
        "time_to_stt_final":    elapsed(t_stt_final),
        "total_time":           round(t_end - t_start, 3) if t_start else -1.0,
    }


async def main(wav_dir: Path, manifest_path: Path, output: Path):
    manifest = load_manifest(manifest_path)
    wav_files = sorted(wav_dir.glob("*.wav"))

    if not wav_files:
        log.error("WAV 파일 없음: %s", wav_dir)
        return

    log.info("총 %d개 파일", len(wav_files))
    rows = []

    for i, wav_path in enumerate(wav_files, 1):
        meta   = manifest.get(wav_path.name, {})
        lang   = meta.get("language_code", "ko")
        ref    = meta.get("reference_text", "")

        log.info("[%d/%d] %s  (lang=%s)", i, len(wav_files), wav_path.name, lang)

        try:
            result = await run_one(wav_path, lang)
        except Exception as e:
            log.error("  실패: %s", e)
            result = {
                "file_name": wav_path.name, "language_code": lang,
                "predicted_text": "", "audio_duration": -1.0,
                "time_to_first_interim": -1.0, "time_to_stt_final": -1.0,
                "total_time": -1.0,
            }

        result["reference_text"] = ref
        rows.append(result)

        log.info("  STT: %s", result["predicted_text"] or "(없음)")
        log.info("  first_interim=%.3fs  stt_final=%.3fs  total=%.3fs",
                 result["time_to_first_interim"],
                 result["time_to_stt_final"],
                 result["total_time"])

    # CSV 저장
    fields = ["file_name", "language_code", "reference_text", "predicted_text",
              "audio_duration", "time_to_first_interim", "time_to_stt_final", "total_time"]
    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    log.info("결과 저장 → %s", output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wav-dir",   type=Path, default=Path("./wav"))
    p.add_argument("--manifest",  type=Path, default=Path("./manifest.csv"))
    p.add_argument("--output",    type=Path, default=Path("./results.csv"))
    args = p.parse_args()

    asyncio.run(main(args.wav_dir, args.manifest, args.output))
