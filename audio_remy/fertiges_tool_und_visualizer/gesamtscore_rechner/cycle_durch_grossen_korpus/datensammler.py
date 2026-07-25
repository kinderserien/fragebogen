import os
import csv
import time

from analyse_main_offen import Config, analyze_file

# -------------KONFIGURATION-------------
AUDIO_ORDNER = "korpus"      # Ordner mit den zu analysierenden Audiodateien
OUTPUT_CSV = "datensatz_activity.csv"

SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")


def sammle_daten(ordner=AUDIO_ORDNER):
    """
    Durchläuft alle unterstützten Audiodateien in 'ordner' und
    berechnet für jede die Activity-Kennzahlen.
    """
    if not os.path.isdir(ordner):
        print(f"[FEHLER] Ordner nicht gefunden: {ordner}")
        return []

    config = Config()  # gleiche Analyse-Parameter (MAX_DURATION, HOP_LENGTH, ...) für alle Dateien

    dateien = sorted(f for f in os.listdir(ordner) if f.lower().endswith(SUPPORTED_EXTENSIONS))

    if not dateien:
        print(f"[WARNUNG] Keine Audiodateien in '{ordner}' gefunden.")
        return []

    ergebnisse = []

    for i, dateiname in enumerate(dateien, start=1):
        pfad = os.path.join(ordner, dateiname)
        print(f"[{i}/{len(dateien)}] Analysiere: {dateiname} ...")

        try:
            res = analyze_file(pfad, config=config, export_json=False)
            ergebnisse.append({
                "titel":                    dateiname,
                "dauer_sekunden":           res["duration_seconds"],
                "onset_count":              res["onset_count"],
                "epm":                      res["events_per_minute"],
                "rms_mean_db":              res["rms_mean_at_onsets"],
                "rms_linear":               res["rms_linear"],
                "activity_score":           res["activity_score"],
                "activity_score_normiert":  res["activity_score_normiert"],
                "unter_pct":                res["unter_pct"],
                "grenze_pct":               res["grenze_pct"],
                "ueber_pct":                res["ueber_pct"],
                "stimme_score":             res["stimme_score"],
                "gesamt_score":             res["gesamt_score"],
            })
        except Exception as e:
            print(f"  [FEHLER] {dateiname} konnte nicht analysiert werden: {e}")

    return ergebnisse


def zeige_tabelle(ergebnisse):
    if not ergebnisse:
        print("Keine Daten zum Anzeigen.")
        return

    header = (f"{'Titel':<35} {'EPM':>8} {'rms_lin':>9} {'Activity':>9} "
               f"{'Act.norm':>9} {'Stimme':>8} {'GESAMT':>8}")
    trennlinie = "-" * len(header)

    print("\n" + trennlinie)
    print(header)
    print(trennlinie)

    for e in ergebnisse:
        print(f"{e['titel']:<35} {e['epm']:>8} {e['rms_linear']:>9} "
              f"{e['activity_score']:>9} {e['activity_score_normiert']:>9} "
              f"{e['stimme_score']:>8} {e['gesamt_score']:>8}")

    print(trennlinie)


def zeige_statistik(ergebnisse):
    if not ergebnisse:
        return

    scores = [e["activity_score"] for e in ergebnisse]
    scores_normiert = [e["activity_score_normiert"] for e in ergebnisse]
    epms = [e["epm"] for e in ergebnisse]
    stimme_scores = [e["stimme_score"] for e in ergebnisse]
    gesamt_scores = [e["gesamt_score"] for e in ergebnisse]

    print("\n---------- KORPUS-STATISTIK ----------")
    print(f"Anzahl Dateien:                   {len(ergebnisse)}")
    print(f"Activity Score (roh) - Min:       {min(scores)}")
    print(f"Activity Score (roh) - Max:       {max(scores)}   <- Kandidat für MAX_EPM-Kalibrierung")
    print(f"Activity Score (roh) - Mittel:    {round(sum(scores) / len(scores), 2)}")
    print(f"Activity Score normiert - Min:    {min(scores_normiert)}")
    print(f"Activity Score normiert - Max:    {max(scores_normiert)}")
    print(f"Activity Score normiert - Mittel: {round(sum(scores_normiert) / len(scores_normiert), 2)}")
    print(f"EPM - Min:                        {min(epms)}")
    print(f"EPM - Max:                        {max(epms)}")
    print(f"EPM - Mittel:                     {round(sum(epms) / len(epms), 2)}")
    print(f"Stimme-Score - Min:               {min(stimme_scores)}")
    print(f"Stimme-Score - Max:               {max(stimme_scores)}")
    print(f"Stimme-Score - Mittel:            {round(sum(stimme_scores) / len(stimme_scores), 2)}")
    print(f"GESAMTSCORE - Min:                {min(gesamt_scores)}")
    print(f"GESAMTSCORE - Max:                {max(gesamt_scores)}")
    print(f"GESAMTSCORE - Mittel:             {round(sum(gesamt_scores) / len(gesamt_scores), 2)}")
    print("---------------------------------------\n")


def speichere_csv(ergebnisse, pfad=OUTPUT_CSV):
    if not ergebnisse:
        return

    with open(pfad, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ergebnisse[0].keys())
        writer.writeheader()
        writer.writerows(ergebnisse)

    print(f"[INFO] Datensatz exportiert nach: {pfad}")


def main():
    ergebnisse = sammle_daten()
    zeige_tabelle(ergebnisse)
    zeige_statistik(ergebnisse)
    speichere_csv(ergebnisse)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(end - start)