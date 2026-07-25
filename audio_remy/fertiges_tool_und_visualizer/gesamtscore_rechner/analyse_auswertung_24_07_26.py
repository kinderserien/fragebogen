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


#-------------AUFMERKSAMKEITS-SCORES-------------
# MAX_EPM: empirisch ermittelter Referenzwert (eigene Kalibrierung über
# mehrere Referenzfolgen), dient zur Normierung von EPM auf 0-100.
MAX_EPM = 275

# Gewichte für die Kombination aus EPM und Lautstärke (rms_linear)
# im normierten Activity-Score. W_EPM > W_RMS, da EPM wichtiger gewichtet wird.
# Muss in Summe 1 ergeben.
W_EPM = 0.7
W_RMS = 0.3




def calc_stimme_score(pitch_results):
    """
    Stimme-Score (0-100):
    50 = neutral (unter_pct == ueber_pct)
    0  = ausschließlich Männerstimmen
    100 = ausschließlich Frauen-/Kinderstimmen
    """
    unter_pct = pitch_results["unter_pct"]
    ueber_pct = pitch_results["ueber_pct"]

    diff = ueber_pct - unter_pct                  # Bereich: -100 bis +100
    stimme_score = round((diff + 100) / 2, 2)     # Bereich: 0 bis 100
    return stimme_score


def calc_activity_score_normiert(onset_results, max_epm=MAX_EPM, w_epm=W_EPM, w_rms=W_RMS):
    """
    Normierter Activity-Score (0-100), gedeckelt bei 100.

    EPM und rms_linear werden GETRENNT auf 0-100 normiert und danach
    gewichtet addiert (statt wie beim rohen activity_score multipliziert).
    Dadurch:
    - bestimmt EPM (w_epm) den Score deutlich stärker als die Lautstärke (w_rms)
    - kann eine leise rms_linear den Score nicht mehr auf ~0 drücken,
      selbst wenn EPM sehr hoch ist
    """
    epm = onset_results["events_per_minute"]
    rms_linear = 10 ** (onset_results["rms_mean_at_onsets"] / 20)

    epm_normiert = min(100, (epm / max_epm) * 100)
    rms_normiert = rms_linear * 100   # rms_linear liegt bereits zwischen 0 und 1

    activity_score_normiert = round(w_epm * epm_normiert + w_rms * rms_normiert, 2)
    return activity_score_normiert


def calc_gesamt_score(stimme_score, activity_score_normiert):
    """
    Gesamtscore (0-100) als Mittelwert aus Stimme- und normiertem Activity-Score.
    >50 = tendenziell aufmerksamkeitsfördernd
    <50 = tendenziell aufmerksamkeitshemmend
    """
    return round((stimme_score + activity_score_normiert)/2, 2)


#-------------OUTPUT-------------
def output():
    onset_results = analyze_onsets(ONSET_JSON)
    pitch_results = analyze_pitch_distribution(PITCH_JSON)

    aktivitäts_score = onset_results['activity_score']
    akt_pro_min = onset_results['events_per_minute']
    unter_pitch = pitch_results['unter_pct']
    grenze_pitch = pitch_results['grenze_pct']
    ueber_pitch = pitch_results['ueber_pct']

    print("\n-----------------------------\n")
    print(f"Activity Score: {aktivitäts_score}")
    print(f"Soundeffekte pro Minute: {akt_pro_min}\n")
    print(f"Unter 180Hz/Männerstimme: {unter_pitch}%")
    print(f"Über 200Hz/Frauenstimme/Kinderstimme: {ueber_pitch}%")
    print(f"Grauzone 180-200Hz: {grenze_pitch}%")
    print("\n-----------------------------\n")

    # -------- Aufmerksamkeits-Scores --------
    stimme_score = calc_stimme_score(pitch_results)
    activity_score_normiert = calc_activity_score_normiert(onset_results)
    gesamt_score = calc_gesamt_score(stimme_score, activity_score_normiert)

    print("---------- AUFMERKSAMKEITS-SCORES ----------\n")
    print(f"Stimme-Score (0-100):            {stimme_score}")
    print(f"Activity-Score normiert (0-100): {activity_score_normiert}")
    print(f"Gesamtscore (0-100):              {gesamt_score}")
    print("\n-----------------------------\n")


if __name__ == "__main__":
    output()