import json
import numpy as np

#-------------ONSET/SOUNDEFFEKTE-------------
ONSET_JSON = "analysis_output/onset.json"

def analyze_onsets(path):
    with open(path) as f:
        data = json.load(f)

    summary    = data["summary"]
    count      = data["onset_count"]
    onsets     = data["onsets"]

    rms_values     = [o["rms_db"] for o in onsets]
    rms_mean       = round(float(np.mean(rms_values)), 4)
    rms_max        = round(float(np.max(rms_values)), 4)
    rms_linear     = 10 ** (rms_mean / 20)
    activity_score = round(summary["events_per_minute"] * rms_linear, 2)

    return {
        "onset_count":        count,
        "events_per_minute":  summary["events_per_minute"],
        "intensity_mean":     summary["intensity_mean"],
        "intensity_max":      summary["intensity_max"],
        "rms_mean_at_onsets": rms_mean,
        "rms_max_at_onsets":  rms_max,
        "activity_score":     activity_score
    }


#-------------PITCH-------------
PITCH_JSON = "analysis_output/pitch.json"
BORDER_LOW  = 180
BORDER_HIGH = 200

def analyze_pitch_distribution(path):
    """
    Analysiert die Verteilung der Grundfrequenz aus pitch.json.

    Kategorien:
    - unter:  pitch_hz < 180        → typisch männlich
    - grenze: 180 <= pitch_hz <= 200 → Grauzone
    - über:   pitch_hz > 200         → typisch weiblich
    - stumm:  pitch_hz == null       → kein Ton erkannt
    """
    with open(path) as f:
        data = json.load(f)

    samples = data["pitch"]
    total = len(samples)

    unter   = 0
    grenze  = 0
    ueber   = 0
    stumm   = 0

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

    voiced = total - stumm

    return {
        "total": total,
        "voiced": voiced,
        "stumm": stumm,
        "unter_pct": round(unter / voiced * 100, 2),
        "grenze_pct": round(grenze / voiced * 100, 2),
        "ueber_pct": round(ueber / voiced * 100, 2)
    }


#-------------OUTPUT-------------
def output():
    onset_results = analyze_onsets(ONSET_JSON)
    pitch_results = analyze_pitch_distribution(PITCH_JSON)

    aktivitäts_score = onset_results['activity_score']
    unter_pitch = pitch_results['unter_pct']
    grenze_pitch = pitch_results['grenze_pct']
    ueber_pitch = pitch_results['ueber_pct']

    print("\n-----------------------------\n")
    print(f"Activity Score: {aktivitäts_score}\n")
    print(f"Unter 180Hz/Männerstimme: {unter_pitch}%")
    print(f"Über 200Hz/Frauenstimme/Kinderstimme: {ueber_pitch}%")
    print(f"Grauzone 180-200Hz: {grenze_pitch}%")
    print("\n-----------------------------\n")


if __name__ == "__main__":
    output()