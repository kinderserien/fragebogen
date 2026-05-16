import numpy as np
import librosa
import soundfile as sf
import json
import os
import Auswertung


class Config:
    AUDIO_FILE = "music_source/peppa_long_tacos.mp3"
    MAX_DURATION = 200
    HOP_LENGTH = 512
    FMIN = 80
    FMAX = 900
    OUTPUT_DIR = "analysis_output"


def load_audio(file_path):
    try:
        y, sr = sf.read(file_path, dtype='float32')
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return y, sr
    except Exception as e:
        print("\n[WARNUNG] SoundFile konnte Datei nicht lesen.\n→ Fallback auf librosa\n")
        y, sr = librosa.load(file_path, sr=None, mono=True)
        return y, sr


def trim_audio(y, sr, max_duration):
    max_samples = int(sr * max_duration)
    return y[:max_samples] if len(y) > max_samples else y


def compute_pitch(y, sr, config):
    f0 = librosa.yin(y, fmin=config.FMIN, fmax=config.FMAX, sr=sr, hop_length=config.HOP_LENGTH)
    t = librosa.times_like(f0, sr=sr, hop_length=config.HOP_LENGTH)
    return f0, t


def compute_rms(y, sr, config):
    rms = librosa.feature.rms(y=y, hop_length=config.HOP_LENGTH)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    t = librosa.times_like(rms, sr=sr, hop_length=config.HOP_LENGTH)
    return rms_db, t


def compute_onsets(y, sr, config, t_rms, rms_db):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=config.HOP_LENGTH)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=config.HOP_LENGTH)
    t_onsets = librosa.frames_to_time(onsets, sr=sr, hop_length=config.HOP_LENGTH)
    intensities = onset_env[onsets]
    rms_at_onsets = np.interp(t_onsets, t_rms, rms_db)
    return t_onsets, intensities, rms_at_onsets


def export_analysis_json(t_onsets, intensities, rms_at_onsets, t_pitch, f0, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)

    duration = float(t_onsets[-1]) if len(t_onsets) > 0 else 0.0
    count = len(t_onsets)

    onset_data = {
        "onset_count": count,
        "duration_seconds": round(duration, 4),
        "summary": {
            "events_per_minute":  round(count / (duration / 60), 2) if duration > 0 else 0,
            "intensity_mean":     round(float(np.mean(intensities)), 4),
            "intensity_max":      round(float(np.max(intensities)), 4),
        },
        "onsets": [
            {
                "index": i,
                "time_seconds": round(float(t), 4),
                "intensity_raw": round(float(v), 4),
                "rms_db": round(float(r), 4)
            }
            for i, (t, v, r) in enumerate(zip(t_onsets, intensities, rms_at_onsets))
        ]
    }

    pitch_data = {
        "sample_count": len(t_pitch),
        "pitch": [
            {
                "index": i,
                "time_seconds": round(float(t), 4),
                "pitch_hz": round(float(h), 4) if not np.isnan(h) else None
            }
            for i, (t, h) in enumerate(zip(t_pitch, f0))
        ]
    }

    files = {
        "onset.json":  onset_data,
        "pitch.json":  pitch_data
    }

    for filename, data in files.items():
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Exportiert → {path}")


def main():
    config = Config()
    y, sr = load_audio(config.AUDIO_FILE)
    y = trim_audio(y, sr, config.MAX_DURATION)

    f0, t_pitch = compute_pitch(y, sr, config)
    rms_db, t_rms = compute_rms(y, sr, config)
    t_onsets, intensities, rms_at_onsets = compute_onsets(y, sr, config, t_rms, rms_db)

    export_analysis_json(t_onsets, intensities, rms_at_onsets, t_pitch, f0, output_dir=config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
    Auswertung.output()
