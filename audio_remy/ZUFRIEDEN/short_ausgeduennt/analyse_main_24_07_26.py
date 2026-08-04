import json
import os
import librosa
import numpy as np
import soundfile as sf
import analyse_auswertung_24_07_26 as Auswertung


class Config:
    AUDIO_FILE = "korpus/gabbys_dollhouse.mp3"
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
    except Exception:
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


def export_analysis_json(t_onsets, intensities, rms_at_onsets, t_pitch, f0, rms_db, onset_env, t_onset_env, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    duration = float(t_onsets[-1]) if len(t_onsets) > 0 else 0.0
    count = len(t_onsets)

    files = {
        "onset.json": {
            "onset_count": count,
            "duration_seconds": round(duration, 4),
            "summary": {
                "events_per_minute": round(count / (duration / 60), 2) if duration > 0 else 0,
                "intensity_mean": round(float(np.mean(intensities)), 4) if count > 0 else 0,
                "intensity_max": round(float(np.max(intensities)), 4) if count > 0 else 0,
            },
            "onsets": [
                {"index": i, "time_seconds": round(float(t), 4), "intensity_raw": round(float(v), 4), "rms_db": round(float(r), 4)}
                for i, (t, v, r) in enumerate(zip(t_onsets, intensities, rms_at_onsets))
            ],
            "envelope": [
                {"time_seconds": round(float(t), 4), "strength": round(float(s), 4)}
                for t, s in zip(t_onset_env, onset_env)
            ]
        },
        "pitch.json": {
            "sample_count": len(t_pitch),
            "pitch": [
                {"index": i, "time_seconds": round(float(t), 4), "pitch_hz": round(float(h), 4) if not np.isnan(h) else None, "rms_db": round(float(r), 4)}
                for i, (t, h, r) in enumerate(zip(t_pitch, f0, rms_db))
            ]
        }
    }

    for filename, data in files.items():
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Exportiert → {path}")


def compute_onset_results(t_onsets, rms_at_onsets, intensities):
    count = len(t_onsets)
    duration = float(t_onsets[-1]) if count > 0 else 0.0

    return {
        "events_per_minute": round(count / (duration / 60), 2) if duration > 0 else 0.0,
        "rms_mean_at_onsets": round(float(np.mean(rms_at_onsets)), 4) if count > 0 else 0.0,
        "intensity_mean": round(float(np.mean(intensities)), 4) if count > 0 else 0.0,
    }


def compute_pitch_results(f0, border_low=Auswertung.BORDER_LOW, border_high=Auswertung.BORDER_HIGH):
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
        return {"unter_pct": 0.0, "grenze_pct": 0.0, "ueber_pct": 0.0}

    return {
        "unter_pct": round(unter / voiced * 100, 2),
        "grenze_pct": round(grenze / voiced * 100, 2),
        "ueber_pct": round(ueber / voiced * 100, 2),
    }


def compute_voice_change_results(f0, t_pitch, rms_db, threshold_semitones=Auswertung.HALBTON_SCHWELLE):
    change_indices = Auswertung.detect_voice_changes(f0, threshold_semitones)
    count = len(change_indices)
    duration = float(t_pitch[-1]) if len(t_pitch) > 0 else 0.0

    return {
        "voice_change_count": count,
        "voice_changes_per_minute": round(count / (duration / 60), 2) if duration > 0 and count > 0 else 0.0,
        "rms_mean_at_changes": round(float(np.mean([rms_db[i] for i in change_indices])), 4) if count > 0 else 0.0,
    }


def analyze_file(audio_file, config=None):
    if config is None:
        config = Config()

    y, sr = load_audio(audio_file)
    y = trim_audio(y, sr, config.MAX_DURATION)

    f0, t_pitch = compute_pitch(y, sr, config)
    rms_db, t_rms = compute_rms(y, sr, config)
    t_onsets, intensities, rms_at_onsets, _, _ = compute_onsets(y, sr, config, t_rms, rms_db)

    onset_results = compute_onset_results(t_onsets, rms_at_onsets, intensities)
    pitch_results = compute_pitch_results(f0)
    voice_change_results = compute_voice_change_results(f0, t_pitch, rms_db)

    stimme_valenz = Auswertung.calc_stimme_score(pitch_results)
    acs = Auswertung.calc_activity_score_normiert(onset_results)
    vcs = Auswertung.calc_vcs(voice_change_results)
    aufmerksamkeits_score = Auswertung.calc_aufmerksamkeits_score(onset_results, pitch_results, voice_change_results)

    return {
        "stimme_valenz": stimme_valenz,
        "activity_score_normiert": acs,
        "voice_change_score": vcs,
        "aufmerksamkeits_score": aufmerksamkeits_score,
        "duration_seconds": round(len(y) / sr, 4),
        "onset_count": len(t_onsets),
        "events_per_minute": onset_results["events_per_minute"],
        "rms_mean_at_onsets": onset_results["rms_mean_at_onsets"],
        "voice_change_count": voice_change_results["voice_change_count"],
        "voice_changes_per_minute": voice_change_results["voice_changes_per_minute"],
        "rms_mean_at_changes": voice_change_results["rms_mean_at_changes"],
        "unter_pct": pitch_results["unter_pct"],
        "grenze_pct": pitch_results["grenze_pct"],
        "ueber_pct": pitch_results["ueber_pct"],
    }


def main():
    config = Config()
    y, sr = load_audio(config.AUDIO_FILE)
    y = trim_audio(y, sr, config.MAX_DURATION)

    f0, t_pitch = compute_pitch(y, sr, config)
    rms_db, t_rms = compute_rms(y, sr, config)
    t_onsets, intensities, rms_at_onsets, onset_env, t_onset_env = compute_onsets(y, sr, config, t_rms, rms_db)

    export_analysis_json(t_onsets, intensities, rms_at_onsets, t_pitch, f0, rms_db, onset_env, t_onset_env, output_dir=config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
    Auswertung.output()