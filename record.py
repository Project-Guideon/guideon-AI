import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000   # Google STT 필수 설정
DURATION    = 5       # 초 (원하는 만큼 변경)
FILENAME    = "wav/ko_04.wav"

print(f"녹음 시작... ({DURATION}초)")
audio = sd.rec(int(DURATION * SAMPLE_RATE),
               samplerate=SAMPLE_RATE,
               channels=1,
               dtype="int16")
sd.wait()
sf.write(FILENAME, audio, SAMPLE_RATE, subtype="PCM_16")
print(f"저장 완료: {FILENAME}")