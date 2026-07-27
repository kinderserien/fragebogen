import os
import csv
import time

from analyse_main_24_07_26 import Config, analyze_file

# -------------KONFIGURATION-------------
AUDIO_ORDNER = "korpus"      # Ordner mit den zu analysierenden Audiodateien
OUTPUT_CSV = "datensatz_aufmerksamkeit.csv"

SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")


def sammle_daten(ordner=AUDIO_ORDNER):
    """
    Durchläuft alle unterstützten Audiodateien in 'ordner' und berechnet
    für jede die vollständigen Kennzahlen aus analyze_file()
    (analyse_main_24_07_26 / analyse_auswertung_24_07_26): sowohl die
    rohen Einzelwerte (EPM, VCPM, Prozentanteile, Dauer, ...) als auch die
    fertigen 0-100-Scores (Stimme-Valenz, ACS, VCS, Gesamt).
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
            res = analyze_file(pfad, config=config)
            ergebnisse.append({
                "titel":                    dateiname,
                "dauer_sekunden":           res["duration_seconds"],
                "onset_count":              res["onset_count"],
                "epm":                      res["events_per_minute"],
                "rms_mean_onsets_db":       res["rms_mean_at_onsets"],
                "unter_pct":                res["unter_pct"],
                "grenze_pct":               res["grenze_pct"],
                "ueber_pct":                res["ueber_pct"],
                "voice_change_count":       res["voice_change_count"],
                "vcpm":                     res["voice_changes_per_minute"],
                "rms_mean_changes_db":      res["rms_mean_at_changes"],
                "stimme_valenz":            res["stimme_valenz"],
                "activity_score_normiert":  res["activity_score_normiert"],
                "voice_change_score":       res["voice_change_score"],
                "aufmerksamkeits_score":    res["aufmerksamkeits_score"],
            })
        except Exception as e:
            print(f"  [FEHLER] {dateiname} konnte nicht analysiert werden: {e}")

    return ergebnisse


def zeige_tabelle(ergebnisse):
    if not ergebnisse:
        print("Keine Daten zum Anzeigen.")
        return

    header = (f"{'Titel':<35} {'EPM':>8} {'VCPM':>8} {'Stimme-V':>9} "
              f"{'ACS':>9} {'VCS':>9} {'GESAMT':>9}")
    trennlinie = "-" * len(header)

    print("\n" + trennlinie)
    print(header)
    print(trennlinie)

    for e in ergebnisse:
        print(f"{e['titel']:<35} {e['epm']:>8} {e['vcpm']:>8} {e['stimme_valenz']:>9} "
              f"{e['activity_score_normiert']:>9} {e['voice_change_score']:>9} "
              f"{e['aufmerksamkeits_score']:>9}")

    print(trennlinie)


def zeige_statistik(ergebnisse):
    if not ergebnisse:
        return

    def stat(feld, kandidat_hinweis=""):
        werte = [e[feld] for e in ergebnisse]
        print(f"{feld:<26} Min: {min(werte):>8}  Max: {max(werte):>8}  "
              f"Mittel: {round(sum(werte) / len(werte), 2):>8}  {kandidat_hinweis}")

    print("\n---------- KORPUS-STATISTIK ----------")
    print(f"Anzahl Dateien: {len(ergebnisse)}\n")
    stat("epm", "<- Kandidat für MAX_EPM-Kalibrierung")
    stat("vcpm", "<- Kandidat für MAX_VCPM-Kalibrierung")
    stat("stimme_valenz")
    stat("activity_score_normiert")
    stat("voice_change_score")
    stat("aufmerksamkeits_score")
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
    print(round(end - start, 2), "Sekunden")