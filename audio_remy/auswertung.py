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
    rms_linear     = 10 ** (rms_mean / 20) #logarithmisch in linear
    activity_score = round(summary["events_per_minute"] * rms_linear, 2)
    """
    print(f"\n{'='*40}")
    print(f"  ONSET VERTEILUNG ANALYSE")
    print(f"{'='*40}")
    print(f"  Ereignisse gesamt:     {count}")
    print(f"  Events pro Minute:     {summary['events_per_minute']}      ← vergleichbar")
    print(f"  Intensität ø (roh):    {summary['intensity_mean']}   ← nur intern")
    print(f"  Intensität max (roh):  {summary['intensity_max']}   ← nur intern")
    print(f"  Weighted Density:      {summary['weighted_density']}  ← nur intern")
    print(f"  RMS ø bei Onsets:      {rms_mean} dB    ← vergleichbar")
    print(f"  RMS max bei Onsets:    {rms_max} dB    ← vergleichbar")
    print(f"  Activity Score:        {activity_score}       ← vergleichbar")
    print(f"{'='*40}\n")
    """
    return {
        "onset_count":        count,
        "events_per_minute":  summary["events_per_minute"],
        "intensity_mean":     summary["intensity_mean"],
        "intensity_max":      summary["intensity_max"],
        "weighted_density":   summary["weighted_density"],
        "rms_mean_at_onsets": rms_mean,
        "rms_max_at_onsets":  rms_max,
        "activity_score":     activity_score
    }


#-------------PITCH-------------
PITCH_JSON = "analysis_output/pitch.json"
BORDER_LOW  = 180   # untere Grenze der Grauzone
BORDER_HIGH = 200   # obere Grenze der Grauzone

def analyze_pitch_distribution(path):
    """
    Analysiert die Verteilung der Grundfrequenz aus pitch.json.

    Kategorien:
    - unter:  pitch_hz < 180        → typisch männlich
    - grenze: 180 <= pitch_hz <= 200 → Grauzone
    - über:   pitch_hz > 200         → typisch weiblich
    - stumm:  pitch_hz == null       → kein Ton erkannt

    Ergebnis in Prozent, bezogen auf alle Samples mit erkanntem Ton
    sowie bezogen auf alle Samples gesamt.
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

    voiced = total - stumm  # Samples mit erkanntem Ton
    """
    print(f"\n{'='*40}")
    print(f"  PITCH VERTEILUNG ANALYSE")
    print(f"{'='*40}")
    print(f"  Gesamt Samples:     {total}")
    print(f"  Davon mit Ton:      {voiced}  ({voiced/total*100:.1f}%)")
    print(f"  Stumm (kein Ton):   {stumm}   ({stumm/total*100:.1f}%)")
    print(f"\n  Bezogen auf alle Samples mit Ton:")
    print(f"  Unter 180 Hz:       {unter/voiced*100:.1f}%")
    print(f"  Grauzone 180-200Hz: {grenze/voiced*100:.1f}%")
    print(f"  Über 200 Hz:        {ueber/voiced*100:.1f}%")
    print(f"{'='*40}\n")
    """
    return {
        "total": total,
        "voiced": voiced,
        "stumm": stumm,
        "unter_pct": round(unter/voiced*100, 2),
        "grenze_pct": round(grenze/voiced*100, 2),
        "ueber_pct":  round(ueber/voiced*100, 2)
    }

aktivitäts_score = analyze_onsets(ONSET_JSON)['activity_score']
events_pro_minute = analyze_onsets(ONSET_JSON)['events_per_minute']

unter_pitch, grenze_pitch, ueber_pitch = (analyze_pitch_distribution(PITCH_JSON)['unter_pct'],
                                          analyze_pitch_distribution(PITCH_JSON)['grenze_pct'],
                                          analyze_pitch_distribution(PITCH_JSON)['ueber_pct'])


def output():
    print("\n-----------------------------\n")
    print(f"Activity Score: {aktivitäts_score}\n")
    print(f"Unter 180Hz/Männerstimme: {unter_pitch}%")
    print(f"Über 200Hz/Frauenstimme/Kinderstimme: {ueber_pitch}%")
    print(f"Grauzone 180-200Hz: {grenze_pitch}%")
    print("\n-----------------------------\n")

if __name__ == "__main__":
    #analyze_pitch_distribution(PITCH_JSON)
    #analyze_onsets(ONSET_JSON)
    output()