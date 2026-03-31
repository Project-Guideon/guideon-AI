import os
import subprocess

INPUT_DIR = "."
OUTPUT_DIR = "wav"
FFMPEG_EXE = r"C:\Users\hjsju\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(".m4a"):
        input_path = os.path.join(INPUT_DIR, filename)
        output_name = os.path.splitext(filename)[0] + ".wav"
        output_path = os.path.join(OUTPUT_DIR, output_name)

        print(f"Converting: {filename} → {output_name}")

        cmd = [
            FFMPEG_EXE,
            "-y",
            "-i", input_path,
            "-ac", "1",
            "-ar", "16000",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ failed: {filename}")
            print(result.stderr)
        else:
            print(f"✅ saved: {output_path}")

print("done")