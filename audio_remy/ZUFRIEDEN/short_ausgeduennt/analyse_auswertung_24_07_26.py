import json
import numpy as np

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


def calc_activity_score_normiert(onset_results, =, w_epm=W_EPM, w_rms=W_RMS):
    epm = onset_results["events_per_minute"]
    rms_linear = 10 ** (onset_results["rms_mean_at_onsets"] / 20)
    epm_normiert = min(100, (epm / ) * 100)
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

    print("\n-----------------------------\n")
    print(f"Activity Score: {onset_results['activity_score']}")
    print(f"Soundeffekte pro Minute: {onset_results['events_per_minute']}\n")
    print(f"Unter 155Hz/Männerstimme: {pitch_results['unter_pct']}%")
    print(f"Über 165Hz/Frauenstimme/Kinderstimme: {pitch_results['ueber_pct']}%")
    print(f"Grauzone 155-165Hz: {pitch_results['grenze_pct']}%")
    print("\n-----------------------------\n")

    stimme_valenz = calc_stimme_score(pitch_results)
    acs = calc_activity_score_normiert(onset_results)
    vcs = calc_vcs(voice_change_results)
    aufmerksamkeits_score = calc_aufmerksamkeits_score(onset_results, pitch_results, voice_change_results)

    print("---------- AUFMERKSAMKEITS-SCORES ----------\n")
    print(f"Stimme-Valenz (0-100, 50=neutral): {stimme_valenz}")
    print(f"Activity-Score normiert (ACS, 0-100): {acs}")
    print(f"Stimmwechsel-Score (VCS, 0-100): {vcs}")
    print(f"Stimmwechsel pro Minute (VCPM): {voice_change_results['voice_changes_per_minute']}")
    print(f"\n>>> GESAMT-AUFMERKSAMKEITSSCORE (0-100): {aufmerksamkeits_score}")
    print("\n-----------------------------\n")


if __name__ == "__main__":
    output()
