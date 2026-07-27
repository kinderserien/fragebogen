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


#-------------STIMMWECHSEL (VCS)-------------
# MAX_VCPM: Referenzwert zur Normierung von VCPM (voice changes per minute)
# auf 0-100 - analog zu MAX_EPM bei den Onsets/ACS.
# ACHTUNG: bislang NICHT empirisch kalibriert (im Gegensatz zu MAX_EPM),
# da noch keine Referenzfolgen für Stimmwechsel ausgewertet wurden.
# Platzhalter, bitte über eigene Kalibrierung anpassen, sobald Referenz-
# werte vorliegen.
MAX_VCPM = 2000

# HALBTON_SCHWELLE: Schwelle für einen "Stimmwechsel"-Punkt. Ändert sich
# die Tonhöhe zwischen zwei aufeinanderfolgenden stimmhaften Frames um
# mehr als HALBTON_SCHWELLE Halbtöne (Semitones), zählt der Frame als
# Stimmwechsel-Punkt (analog zu den Onset-Punkten bei ACS).
# Halbton-Differenz = 12 * log2(f2/f1).
HALBTON_SCHWELLE = 1.0

# Gewichte für die Kombination aus VCPM und Lautstärke (rms_linear) im
# VCS - genau wie W_EPM/W_RMS beim ACS. Muss in Summe 1 ergeben.
W_VCPM   = 0.7
W_RMS_VC = 0.3


def detect_voice_changes(pitch_hz_list, threshold_semitones=HALBTON_SCHWELLE):
    """
    Erkennt "Stimmwechsel"-Punkte: Frames, bei denen sich die Tonhöhe zum
    vorherigen stimmhaften Frame um mehr als threshold_semitones Halbtöne
    ändert (Halbton-Differenz = 12 * log2(f2/f1)).

    Stumme Frames (None/NaN) werden übersprungen - es wird jeweils mit dem
    letzten stimmhaften Frame verglichen, nicht zwingend dem direkten
    Vorgänger im Array.

    Gibt die Liste der Indizes zurück, an denen ein Stimmwechsel-Punkt
    erkannt wurde - analog zu den Onset-Indizes bei ACS.
    """
    change_indices = []
    prev_hz = None

    for i, hz in enumerate(pitch_hz_list):
        if hz is None or (isinstance(hz, float) and np.isnan(hz)):
            continue
        if prev_hz is not None and prev_hz > 0 and hz > 0:
            semitone_diff = abs(12 * np.log2(hz / prev_hz))
            if semitone_diff > threshold_semitones:
                change_indices.append(i)
        prev_hz = hz

    return change_indices


def analyze_voice_changes(path, threshold_semitones=HALBTON_SCHWELLE):
    """
    Analysiert Stimmwechsel aus pitch.json (reine JSON-Route, analog zu
    analyze_onsets() für die Onsets). Liefert VCPM (voice changes per
    minute) sowie die mittlere RMS an den Stimmwechsel-Punkten - das ist
    genau das, was calc_vcs() braucht.
    """
    with open(path) as f:
        data = json.load(f)

    samples = data["pitch"]
    pitch_hz_list = [s["pitch_hz"] for s in samples]
    rms_db_list   = [s["rms_db"] for s in samples]
    times         = [s["time_seconds"] for s in samples]

    change_indices = detect_voice_changes(pitch_hz_list, threshold_semitones)
    count = len(change_indices)

    duration = times[-1] if times else 0.0
    voice_changes_per_minute = round(count / (duration / 60), 2) if duration > 0 and count > 0 else 0.0

    if count > 0:
        rms_at_changes = [rms_db_list[i] for i in change_indices]
        rms_mean_at_changes = round(float(np.mean(rms_at_changes)), 4)
    else:
        rms_mean_at_changes = 0.0

    return {
        "voice_change_count":       count,
        "voice_changes_per_minute": voice_changes_per_minute,
        "rms_mean_at_changes":      rms_mean_at_changes,
    }


def calc_vcs(voice_change_results, max_vcpm=MAX_VCPM, w_vcpm=W_VCPM, w_rms=W_RMS_VC):
    """
    Stimmwechsel-Score (VCS, 0-100), EINPOLIG, gedeckelt bei 100.

    Genau wie beim ACS (calc_activity_score_normiert): VCPM und rms_linear
    (hier an den Stimmwechsel-Punkten statt an Onsets gemessen) werden
    GETRENNT auf 0-100 normiert und danach gewichtet addiert, statt wie
    beim rohen activity_score multipliziert.
    """
    vcpm = voice_change_results["voice_changes_per_minute"]
    rms_linear = 10 ** (voice_change_results["rms_mean_at_changes"] / 20)

    vcpm_normiert = min(100, (vcpm / max_vcpm) * 100)
    rms_normiert  = rms_linear * 100   # rms_linear liegt bereits zwischen 0 und 1

    vcs = round(w_vcpm * vcpm_normiert + w_rms * rms_normiert, 2)
    return vcs


#-------------AUFMERKSAMKEITS-SCORES-------------
# MAX_EPM: empirisch ermittelter Referenzwert (eigene Kalibrierung über
# mehrere Referenzfolgen), dient zur Normierung von EPM auf 0-100.
MAX_EPM = 275


# Gewichte für die Kombination aus EPM und Lautstärke (rms_linear) im
# normierten Activity-Score (ACS). W_EPM > W_RMS, da EPM wichtiger
# gewichtet wird. Muss in Summe 1 ergeben.
W_EPM = 0.7
W_RMS = 0.3

# Gewichte für den finalen Gesamt-Aufmerksamkeitsscore aus GENAU 3 Achsen:
# Stimme-Valenz, Activity-Score normiert (ACS), Stimmwechsel-Score (VCS).
# Muss in Summe 1 ergeben.
W_STIMME_VALENZ = 2/12
W_ACS           = 4/12
W_VCS           = 6/12

# ALT: Gewichte des vorherigen 3-Achsen-Modells (Stimme-Valenz,
# Ereignisintensität, Lautstärke-Mittelwert). Durch ACS/VCS oben ersetzt.
# W_EREIGNISINTENSITAET  = 1/3
# W_LAUTSTAERKE_MITTEL   = 1/3

# STIMME_GAIN: Verstärkungsfaktor für die Stimme-Valenz (siehe calc_stimme_score).
# score = 50 + STIMME_GAIN * (ueber_pct - unter_pct), auf 0-100 geklippt.
# GAIN=0.5 -> alte Formel: score = 50 + diff/2 (voller diff-Bereich -100..100
#             wird auf 0..100 abgebildet, dadurch in der Praxis gestaucht,
#             da diff wegen der Grauzone selten ±100 erreicht)
# GAIN=1.0 -> score = 50 + diff (z.B. 70/30 -> 90, 60/40 -> 70). Nutzt die
#             Skala bei typischen, moderaten Verhältnissen besser aus,
#             sättigt dafür schon ab |diff| > 50 (z.B. 75/25) bei 0/100.
STIMME_GAIN = 1.0


def calc_stimme_score(pitch_results, gain=STIMME_GAIN):
    """
    Stimme-Valenz (0-100), ZWEIPOLIG:
    50 = neutral (unter_pct == ueber_pct)
    0  = ausschließlich Männerstimmen
    100 = ausschließlich Frauen-/Kinderstimmen

    score = 50 + gain * (ueber_pct - unter_pct), auf 0-100 geklippt.

    gain=0.5 entspricht der ursprünglichen Formel (50 + diff/2) und bildet
    den vollen theoretischen diff-Bereich (-100..100) linear auf 0-100 ab -
    dadurch in der Praxis gestaucht, weil diff wegen der Grauzone selten
    ±100 erreicht.
    gain=1.0 (Standard) nutzt die Skala bei typischen/moderaten Verhält-
    nissen (60/40 -> 70, 70/30 -> 90) sichtbarer aus, sättigt dafür schon
    ab |diff| > 50 (z.B. 75/25) an den Rändern - das ist gewollt, siehe
    STIMME_GAIN oben.
    """
    unter_pct = pitch_results["unter_pct"]
    ueber_pct = pitch_results["ueber_pct"]

    diff = ueber_pct - unter_pct                  # Bereich: -100 bis +100
    stimme_score = 50 + gain * diff
    stimme_score = float(np.clip(stimme_score, 0, 100))
    return round(stimme_score, 2)


def calc_activity_score_normiert(onset_results, max_epm=MAX_EPM, w_epm=W_EPM, w_rms=W_RMS):
    """
    Normierter Activity-Score (ACS, 0-100), gedeckelt bei 100.

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


# ALT / nicht mehr Teil des Gesamtscores. Inhaltlich identisch zu
# calc_activity_score_normiert() - wurde durch dessen Nutzung als offizielle
# ACS-Komponente im Gesamtscore redundant. Auskommentiert, siehe
# calc_activity_score_normiert() oben.
# def calc_ereignisintensitaet_norm(onset_results, max_epm=MAX_EPM, w_epm=W_EPM, w_rms=W_RMS):
#     """
#     Normierte Ereignisintensität (0-100), gedeckelt bei 100.
#
#     EPM und rms_linear werden GETRENNT auf 0-100 normiert (EPM über den
#     kalibrierten Referenzwert MAX_EPM, rms_linear liegt schon 0-1) und
#     danach gewichtet addiert - genau wie bei calc_activity_score_normiert().
#     Kein rohes Produkt mehr, da MAX_ACTIVITY_RAW nie empirisch ermittelt
#     wurde und für ein unkalibriertes Produkt kein sinnvoller Referenzwert
#     vorliegt.
#     """
#     epm = onset_results["events_per_minute"]
#     rms_linear = 10 ** (onset_results["rms_mean_at_onsets"] / 20)
#
#     epm_normiert = min(100, (epm / max_epm) * 100)
#     rms_normiert = rms_linear * 100   # rms_linear liegt bereits zwischen 0 und 1
#
#     return round(w_epm * epm_normiert + w_rms * rms_normiert, 2)


# ALT / beibehalten nur als Referenz. Mittelte einen einpoligen Score
# (Activity) direkt mit einem zweipoligen Score (Stimme, 50=neutral) - das
# vermischt Arousal und Valenz und ist inhaltlich problematisch. Durch
# calc_aufmerksamkeits_score() ersetzt, nicht mehr im Einsatz.
# def calc_gesamt_score(stimme_score, activity_score_normiert):
#     return round((stimme_score + activity_score_normiert)/2, 2)


# ALT / nicht mehr Teil des Gesamtscores, da Komponente "Stimme-Intensität"
# nicht mehr gebraucht wird (durch VCS als eigene Achse ersetzt). Funktion
# bleibt zu Referenzzwecken auskommentiert erhalten.
# def calc_stimme_intensitaet(pitch_results):
#     """
#     Stimm-Intensität (0-100), EINPOLIG (0=nur neutrale Grauzone, 100=nie
#     in der Grauzone).
#
#     Beschreibt, welcher Anteil der Sprachzeit AUSSERHALB der neutralen
#     180-200 Hz-Zone liegt - unabhängig davon, ob nach unten (tief/männlich)
#     oder oben (hoch/kindlich) ausgeschlagen wird.
#
#     Bewusst NICHT als abs(stimme_score - 50)*2 berechnet: das wäre der
#     Netto-Ausschlag zwischen den Polen und würde z.B. bei 50% sehr tiefer
#     und 50% sehr hoher Stimme (= akustisch sehr auffällig) fälschlich 0
#     ergeben, weil sich die beiden Extreme gegenseitig aufheben würden.
#     unter_pct + ueber_pct vermeidet dieses Canceling.
#     """
#     unter_pct = pitch_results["unter_pct"]
#     ueber_pct = pitch_results["ueber_pct"]
#     return round(unter_pct + ueber_pct, 2)   # = 100 - grenze_pct


# ALT / bereits vor dieser Änderung ungenutzt (wurde in output() nirgends
# aufgerufen - calc_activity_score_normiert() berechnet rms_linear intern
# selbst). Auskommentiert.
# def calc_lautstaerke_mittelwert(rms_db):
#     """
#     Lautstärke-Mittelwert (0-100), EINPOLIG.
#
#     rms_db ist die volle RMS-Kurve über die GESAMTE Episode (nicht nur an
#     Onset-Punkten wie bei calc_lautstaerke_norm). rms_db ist bereits relativ
#     zum Maximum des Tracks normiert (amplitude_to_db mit ref=np.max), liegt
#     also immer <= 0 dB. Pro Frame in linear umrechnen (0-1) und danach den
#     Mittelwert bilden - NICHT den Mittelwert der dB-Werte selbst nehmen und
#     dann erst umrechnen, das wäre wegen der Log-Skala nicht dasselbe.
#     """
#     rms_linear_all = 10 ** (np.asarray(rms_db) / 20)
#     mittelwert = float(np.mean(rms_linear_all)) * 100   # rms_linear liegt 0-1
#     return round(min(100.0, mittelwert), 2)


# ALT / bereits vor dieser Änderung ungenutzt. Auskommentiert.
# def calc_lautstaerke_norm(onset_results):
#     """
#     Normierte Lautstärke (0-100) aus rms_mean_at_onsets (dB).
#     rms_linear liegt bereits zwischen 0 und 1, daher *100 ausreichend.
#
#     Hinweis: das ist die Lautstärke AN DEN ONSETS, nicht zwingend die
#     globale Lautstärke des gesamten Clips. Falls zusätzlich eine
#     track-weite RMS-Kennzahl vorliegt, diese hier stattdessen einsetzen.
#     """
#     rms_linear = 10 ** (onset_results["rms_mean_at_onsets"] / 20)
#     return round(rms_linear * 100, 2)


# ALT / bereits vor dieser Änderung ungenutzt. Auskommentiert.
# def calc_effektrate_norm(onset_results, max_epm=MAX_EPM):
#     """Normierte Effektrate (0-100), gedeckelt bei 100."""
#     epm = onset_results["events_per_minute"]
#     return round(min(100, (epm / max_epm) * 100), 2)


def calc_aufmerksamkeits_score(onset_results, pitch_results, voice_change_results,
                                w_stimme_valenz=W_STIMME_VALENZ,
                                w_acs=W_ACS,
                                w_vcs=W_VCS):
    """
    Gesamt-Aufmerksamkeitsscore (0-100) aus GENAU 3 Achsen:

    1) Stimme-Valenz:              50 + GAIN*(ueber_pct - unter_pct),
                                    geklippt (ZWEIPOLIG, 50=neutral)
    2) Activity-Score normiert (ACS): EPM (Onsets/Soundeffekte pro Minute)
                                    kombiniert mit rms_linear an den
                                    Onsets, normiert (EINPOLIG)
    3) Stimmwechsel-Score (VCS):   VCPM (Stimmwechsel pro Minute, Frames
                                    mit Tonhöhenänderung > 1 Halbton)
                                    kombiniert mit rms_linear an den
                                    Stimmwechsel-Punkten, normiert
                                    (EINPOLIG)

    ACHTUNG: Komponente 1 ist ZWEIPOLIG (50=neutral, 0/100=Extreme),
    Komponente 2 und 3 sind EINPOLIG (0=nichts, 100=maximal). Genau diese
    Mischung war zuvor schon Bestandteil des Scores (dort mit Lautstärke-
    Mittelwert statt VCS als dritter Achse) - hier bewusst beibehalten.
    """
    stimme_valenz = calc_stimme_score(pitch_results)
    acs = calc_activity_score_normiert(onset_results)
    vcs = calc_vcs(voice_change_results)

    aufmerksamkeits_score = (
        w_stimme_valenz * stimme_valenz +
        w_acs * acs +
        w_vcs * vcs
    )
    return round(aufmerksamkeits_score, 2)


# ALT / bereits vor dieser Änderung nur für calc_lautstaerke_mittelwert()
# in der reinen JSON-Route gebraucht - da diese Komponente nicht mehr Teil
# des Scores ist, wird auch extract_rms_db_track() nicht mehr benötigt.
# def extract_rms_db_track(path=PITCH_JSON):
#     """
#     Liest die volle RMS-Kurve über die gesamte Episode aus pitch.json
#     (dort liegt rms_db pro Sample bereits vor) - wurde für
#     calc_lautstaerke_mittelwert() in der reinen JSON-Route (output())
#     gebraucht, da analyze_onsets() nur onset-bezogene RMS-Werte liefert.
#     """
#     with open(path) as f:
#         data = json.load(f)
#     return [s["rms_db"] for s in data["pitch"]]


#-------------OUTPUT-------------
def output():
    onset_results = analyze_onsets(ONSET_JSON)
    pitch_results = analyze_pitch_distribution(PITCH_JSON)
    voice_change_results = analyze_voice_changes(PITCH_JSON)

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
    stimme_valenz = calc_stimme_score(pitch_results)
    # stimme_intensitaet = calc_stimme_intensitaet(pitch_results)  # nicht mehr Teil des Scores
    acs = calc_activity_score_normiert(onset_results)
    vcs = calc_vcs(voice_change_results)
    # lautstaerke_mittelwert = calc_lautstaerke_mittelwert(extract_rms_db_track())  # nicht mehr Teil des Scores
    aufmerksamkeits_score = calc_aufmerksamkeits_score(
        onset_results, pitch_results, voice_change_results
    )

    print("---------- AUFMERKSAMKEITS-SCORES ----------\n")
    print(f"Stimme-Valenz (0-100, 50=neutral, Richtung tief<->hoch): {stimme_valenz}")
    print(f"Activity-Score normiert (ACS, 0-100):                    {acs}")
    print(f"Stimmwechsel-Score (VCS, 0-100):                         {vcs}")
    print(f"Stimmwechsel pro Minute (VCPM, nur Info):                {voice_change_results['voice_changes_per_minute']}")
    print(f"\n>>> GESAMT-AUFMERKSAMKEITSSCORE (0-100): {aufmerksamkeits_score}")
    print("\n-----------------------------\n")


if __name__ == "__main__":
    output()
