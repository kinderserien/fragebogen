import numpy as np
import librosa
import soundfile as sf
import json
import os
import analyse_auswertung_23_07_26 as Auswertung  # Importiert Auswertung_neu.py


class Config:
    AUDIO_FILE = "korpus/paw_patrol.mp3"
    MAX_DURATION = 1500
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
    t_onset_env = librosa.times_like(onset_env, sr=sr, hop_length=config.HOP_LENGTH)

    intensities = onset_env[onsets]
    rms_at_onsets = np.interp(t_onsets, t_rms, rms_db)

    return t_onsets, intensities, rms_at_onsets, onset_env, t_onset_env


def export_analysis_json(t_onsets, intensities, rms_at_onsets, t_pitch, f0, rms_db, onset_env, t_onset_env,
                         output_dir="."):
    os.makedirs(output_dir, exist_ok=True)

    duration = float(t_onsets[-1]) if len(t_onsets) > 0 else 0.0
    count = len(t_onsets)

    onset_data = {
        "onset_count": count,
        "duration_seconds": round(duration, 4),
        "summary": {
            "events_per_minute": round(count / (duration / 60), 2) if duration > 0 else 0,
            "intensity_mean": round(float(np.mean(intensities)), 4) if count > 0 else 0,
            "intensity_max": round(float(np.max(intensities)), 4) if count > 0 else 0,
        },
        "onsets": [
            {
                "index": i,
                "time_seconds": round(float(t), 4),
                "intensity_raw": round(float(v), 4),
                "rms_db": round(float(r), 4)
            }
            for i, (t, v, r) in enumerate(zip(t_onsets, intensities, rms_at_onsets))
        ],
        "envelope": [
            {
                "time_seconds": round(float(t), 4),
                "strength": round(float(s), 4)
            }
            for t, s in zip(t_onset_env, onset_env)
        ]
    }

    pitch_data = {
        "sample_count": len(t_pitch),
        "pitch": [
            {
                "index": i,
                "time_seconds": round(float(t), 4),
                "pitch_hz": round(float(h), 4) if not np.isnan(h) else None,
                "rms_db": round(float(r), 4)
            }
            for i, (t, h, r) in enumerate(zip(t_pitch, f0, rms_db))
        ]
    }

    files = {
        "onset.json": onset_data,
        "pitch.json": pitch_data
    }

    for filename, data in files.items():
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Exportiert → {path}")


def compute_onset_results(t_onsets, rms_at_onsets):
    """
    Liefert die Kennzahlen, die calc_activity_score_normiert() aus
    Auswertung braucht (events_per_minute, rms_mean_at_onsets) -
    direkt aus dem RAM, ohne JSON-Umweg.
    """
    count = len(t_onsets)
    duration = float(t_onsets[-1]) if count > 0 else 0.0

    events_per_minute = round(count / (duration / 60), 2) if duration > 0 else 0.0
    rms_mean_at_onsets = round(float(np.mean(rms_at_onsets)), 4) if count > 0 else 0.0

    return {
        "events_per_minute": events_per_minute,
        "rms_mean_at_onsets": rms_mean_at_onsets,
    }


def compute_pitch_results(f0, border_low=Auswertung.BORDER_LOW, border_high=Auswertung.BORDER_HIGH):
    """
    Liefert die Kennzahlen, die calc_stimme_score() aus Auswertung
    braucht (unter_pct, ueber_pct) - direkt aus dem RAM, ohne JSON-Umweg.
    """
    total = len(f0)
    unter = grenze = ueber = stumm = 0

    for hz in f0:
        if np.isnan(hz):
            stumm += 1
        elif hz < border_low:
            unter += 1
        elif hz > border_high:
            ueber += 1
        else:
            grenze += 1

    voiced = total - stumm

    if voiced == 0:
        unter_pct = grenze_pct = ueber_pct = 0.0
    else:
        unter_pct = round(unter / voiced * 100, 2)
        grenze_pct = round(grenze / voiced * 100, 2)
        ueber_pct = round(ueber / voiced * 100, 2)

    return {
        "unter_pct": unter_pct,
        "grenze_pct": grenze_pct,
        "ueber_pct": ueber_pct,
    }


def analyze_file(audio_file, config=None):
    """
    Führt die komplette Analyse für EINE Audiodatei aus (ohne JSON-Export)
    und gibt die drei zentralen Aufmerksamkeits-Scores zurück.

    Wird von cycle.py genutzt, um einen ganzen Ordner voller Dateien
    durchzugehen, ohne dass für jede Datei zwischen Skripten/JSON hin-
    und hergesprungen werden muss.
    """
    if config is None:
        config = Config()

    y, sr = load_audio(audio_file)
    y = trim_audio(y, sr, config.MAX_DURATION)

    f0, t_pitch = compute_pitch(y, sr, config)
    rms_db, t_rms = compute_rms(y, sr, config)
    t_onsets, intensities, rms_at_onsets, onset_env, t_onset_env = compute_onsets(y, sr, config, t_rms, rms_db)

    onset_results = compute_onset_results(t_onsets, rms_at_onsets)
    pitch_results = compute_pitch_results(f0)

    stimme_score = Auswertung.calc_stimme_score(pitch_results)
    activity_score_normiert = Auswertung.calc_activity_score_normiert(onset_results)
    gesamt_score = Auswertung.calc_gesamt_score(stimme_score, activity_score_normiert)

    return {
        "stimme_score": stimme_score,
        "activity_score_normiert": activity_score_normiert,
        "gesamt_score": gesamt_score,
    }


def main():
    config = Config()
    y, sr = load_audio(config.AUDIO_FILE)
    y = trim_audio(y, sr, config.MAX_DURATION)

    f0, t_pitch = compute_pitch(y, sr, config)
    rms_db, t_rms = compute_rms(y, sr, config)
    t_onsets, intensities, rms_at_onsets, onset_env, t_onset_env = compute_onsets(y, sr, config, t_rms, rms_db)

    export_analysis_json(
        t_onsets, intensities, rms_at_onsets,
        t_pitch, f0, rms_db,
        onset_env, t_onset_env,
        output_dir=config.OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
    Auswertung.output()