import json
import os
import librosa
import numpy as np
import soundfile as sf


# ============================================================
# KONSTANTEN / GEWICHTE
# ============================================================
ONSET_JSON = "analysis_output/onset.json"
PITCH_JSON = "analysis_output/pitch.json"

BORDER_LOW = 155.0
BORDER_HIGH = 165.0
MAX_VCPM = 2000
HALBTON_SCHWELLE = 1.0
MAX_EPM = 300

W_VCPM = 0.7
W_RMS_VC = 0.3
W_EPM = 0.7
W_RMS = 0.3

W_STIMME_VALENZ = 2 / 12
W_ACS = 4 / 12
W_VCS = 6 / 12


class Config:
    AUDIO_FILE = "korpus/peppa_zoo.mp3"
    MAX_DURATION = 1500
    HOP_LENGTH = 512
    FMIN = 80
    FMAX = 900
    OUTPUT_DIR = "analysis_output"


# ============================================================
# AUSWERTUNG (JSON-basiert, für die Einzeldatei-Ausgabe via output())
# ============================================================
def analyze_onsets(path):
    with open(path) as f:
        data = json.load(f)

    summary = data["summary"]
    rms_values = [o["rms_db"] for o in data["onsets"]]
    rms_mean = round(float(np.mean(rms_values)), 4) if rms_values else 0.0
    rms_linear = 10 ** (rms_mean / 20)

    return {
        "onset_count": data["onset_count"],
        "events_per_minute": summary["events_per_minute"],
        "intensity_mean": summary["intensity_mean"],
        "intensity_max": summary["intensity_max"],
        "rms_mean_at_onsets": rms_mean,
        "rms_max_at_onsets": round(float(np.max(rms_values)), 4) if rms_values else 0.0,
        "activity_score": round(summary["events_per_minute"] * rms_linear, 2)
    }


def analyze_pitch_distribution(path):
    with open(path) as f:
        data = json.load(f)

    samples = data["pitch"]
    unter = grenze = ueber = stumm = 0

    for s in samples:
        hz = s["pitch_hz"]
        if hz is None:
            stumm += 1
        elif hz < BORDER_LOW:
            unter += 1
        elif hz > BORDER_HIGH:
            ueber += 1
        else:
            grenze += 1

    voiced = len(samples) - stumm
    if voiced == 0:
        return {"total": len(samples), "voiced": 0, "stumm": stumm, "unter_pct": 0.0, "grenze_pct": 0.0, "ueber_pct": 0.0}

    return {
        "total": len(samples),
        "voiced": voiced,
        "stumm": stumm,
        "unter_pct": round(unter / voiced * 100, 2),
        "grenze_pct": round(grenze / voiced * 100, 2),
        "ueber_pct": round(ueber / voiced * 100, 2)
    }


def detect_voice_changes(pitch_hz_list, threshold_semitones=HALBTON_SCHWELLE):
    change_indices = []
    prev_hz = None

    for i, hz in enumerate(pitch_hz_list):
        if hz is None or np.isnan(hz):
            continue
        if prev_hz is not None and prev_hz > 0 and hz > 0:
            if abs(12 * np.log2(hz / prev_hz)) > threshold_semitones:
                change_indices.append(i)
        prev_hz = hz

    return change_indices


def analyze_voice_changes(path, threshold_semitones=HALBTON_SCHWELLE):
    with open(path) as f:
        data = json.load(f)

    samples = data["pitch"]
    pitch_hz_list = [s["pitch_hz"] for s in samples]
    change_indices = detect_voice_changes(pitch_hz_list, threshold_semitones)
    count = len(change_indices)

    duration = samples[-1]["time_seconds"] if samples else 0.0
    vcpm = round(count / (duration / 60), 2) if duration > 0 and count > 0 else 0.0

    rms_mean_at_changes = round(float(np.mean([samples[i]["rms_db"] for i in change_indices])), 4) if count > 0 else 0.0

    return {
        "voice_change_count": count,
        "voice_changes_per_minute": vcpm,
        "rms_mean_at_changes": rms_mean_at_changes,
    }


def calc_vcs(voice_change_results, max_vcpm=MAX_VCPM, w_vcpm=W_VCPM, w_rms=W_RMS_VC):
    vcpm = voice_change_results["voice_changes_per_minute"]
    rms_linear = 10 ** (voice_change_results["rms_mean_at_changes"] / 20)
    vcpm_normiert = min(100, (vcpm / max_vcpm) * 100)
    return round(w_vcpm * vcpm_normiert + w_rms * (rms_linear * 100), 2)


def calc_stimme_score(pitch_results):
    diff = pitch_results["ueber_pct"] - pitch_results["unter_pct"]
    return round(float(np.clip(50 + diff, 0, 100)), 2)


def calc_activity_score_normiert(onset_results, max_epm=MAX_EPM, w_epm=W_EPM, w_rms=W_RMS):
    epm = onset_results["events_per_minute"]
    rms_linear = 10 ** (onset_results["rms_mean_at_onsets"] / 20)
    epm_normiert = min(100, (epm / max_epm) * 100)
    return round(w_epm * epm_normiert + w_rms * (rms_linear * 100), 2)


def calc_aufmerksamkeits_score(onset_results, pitch_results, voice_change_results):
    stimme_valenz = calc_stimme_score(pitch_results)
    acs = calc_activity_score_normiert(onset_results)
    vcs = calc_vcs(voice_change_results)
    return round(W_STIMME_VALENZ * stimme_valenz + W_ACS * acs + W_VCS * vcs, 2)


def output():
    onset_results = analyze_onsets(ONSET_JSON)
    pitch_results = analyze_pitch_distribution(PITCH_JSON)
    voice_change_results = analyze_voice_changes(PITCH_JSON)

    stimme_valenz = calc_stimme_score(pitch_results)
    acs = calc_activity_score_normiert(onset_results)
    vcs = calc_vcs(voice_change_results)
    aufmerksamkeits_score = calc_aufmerksamkeits_score(onset_results, pitch_results, voice_change_results)

    rms_linear_onsets = round(10 ** (onset_results["rms_mean_at_onsets"] / 20), 4)
    rms_linear_changes = round(10 ** (voice_change_results["rms_mean_at_changes"] / 20), 4)

    zeilen = [
        ("EPM (Events/Min, Onsets)",                  onset_results["events_per_minute"]),
        ("RMS an Onsets (dB)",                         onset_results["rms_mean_at_onsets"]),
        ("RMS-linear an Onsets (0-1, → ACS)",          rms_linear_onsets),
        ("ACS – Activity-Score normiert",              acs),
        ("",                                           None),
        ("VCPM (Stimmwechsel/Min)",                    voice_change_results["voice_changes_per_minute"]),
        ("RMS an Stimmwechseln (dB)",                  voice_change_results["rms_mean_at_changes"]),
        ("RMS-linear an Stimmwechseln (0-1, → VCS)",   rms_linear_changes),
        ("VCS – Stimmwechsel-Score normiert",          vcs),
        ("",                                     None),
        ("Pitch unter 155Hz (Männerstimme)",     f"{pitch_results['unter_pct']}%"),
        ("Pitch Grauzone 155-165Hz",             f"{pitch_results['grenze_pct']}%"),
        ("Pitch über 165Hz (Frauen/Kinder)",     f"{pitch_results['ueber_pct']}%"),
        ("Stimme-Valenz (0-100, 50=neutral)",    stimme_valenz),
    ]

    label_breite = max(len(l) for l, _ in zeilen if l)

    print("\n---------- AUFMERKSAMKEITS-ANALYSE ----------\n")
    for label, wert in zeilen:
        if not label:
            print()
            continue
        print(f"{label:<{label_breite}} : {wert}")
    print(f"\n{'>>> GESAMT-AUFMERKSAMKEITSSCORE (0-100)':<{label_breite}} : {aufmerksamkeits_score}")
    print("\n----------------------------------------------\n")


# ============================================================
# AUDIO-EXTRAKTION / PIPELINE (RAM-basiert, für analyze_file() / main())
# ============================================================
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


def compute_pitch_results(f0, border_low=BORDER_LOW, border_high=BORDER_HIGH):
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


def compute_voice_change_results(f0, t_pitch, rms_db, threshold_semitones=HALBTON_SCHWELLE):
    change_indices = detect_voice_changes(f0, threshold_semitones)
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

    stimme_valenz = calc_stimme_score(pitch_results)
    acs = calc_activity_score_normiert(onset_results)
    vcs = calc_vcs(voice_change_results)
    aufmerksamkeits_score = calc_aufmerksamkeits_score(onset_results, pitch_results, voice_change_results)

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
    output()