import os
import shutil
import subprocess


def add_silence_to_audio_ffmpeg(audio_path, tmp_audio_path, silence_duration_s=0.5):
    silence_duration_s = float(silence_duration_s)
    if silence_duration_s < 0:
        raise ValueError("silence_duration_s must be non-negative")
    os.makedirs(os.path.dirname(os.path.abspath(tmp_audio_path)), exist_ok=True)
    if silence_duration_s == 0:
        if os.path.abspath(audio_path) != os.path.abspath(tmp_audio_path):
            shutil.copy2(audio_path, tmp_audio_path)
        return tmp_audio_path

    delay_ms = round(silence_duration_s * 1000)
    command = [
        "ffmpeg",
        "-v", "error",
        "-y",
        "-i", audio_path,
        "-af", f"adelay={delay_ms}:all=1",
        tmp_audio_path,
    ]

    subprocess.run(command, check=True)
    return tmp_audio_path
