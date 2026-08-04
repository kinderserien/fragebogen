import csv
import os
import time
from analyse_main_24_07_26 import Config, analyze_file

AUDIO_ORDNER = "korpus"
OUTPUT_CSV = "datensatz_aufmerksamkeit.csv"
SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")


def sammle_daten(ordner=AUDIO_ORDNER):
    if not os.path.isdir(ordner):
        print(f"[FEHLER] Ordner nicht gefunden: {ordner}")
        return []

    config = Config()
    dateien = sorted(f for f in os.listdir(ordner) if f.lower().endswith(SUPPORTED_EXTENSIONS))

    if not dateien:
        print(f"[WARNUNG] Keine Audiodateien in '{ordner}' gefunden.")
        return []

    ergebnisse = []
    for i, dateiname in enumerate(dateien, start=1):
        print(f"[{i}/{len(dateien)}] Analysiere: {dateiname} ...")
        try:
            res = analyze_file(os.path.join(ordner, dateiname), config=config)
            ergebnisse.append({"titel": dateiname, **res})
        except Exception as e:
            print(f"  [FEHLER] {dateiname} konnte nicht analysiert werden: {e}")

    return ergebnisse


def zeige_tabelle(ergebnisse):
    if not ergebnisse:
        print("Keine Daten zum Anzeigen.")
        return

    header = f"{'Titel':<35} {'EPM':>8} {'VCPM':>8} {'Stimme-V':>9} {'ACS':>9} {'VCS':>9} {'GESAMT':>9}"
    trennlinie = "-" * len(header)

    print("\n" + trennlinie)
    print(header)
    print(trennlinie)

    for e in ergebnisse:
        print(f"{e['titel']:<35} {e['events_per_minute']:>8} {e['voice_changes_per_minute']:>8} {e['stimme_valenz']:>9} "
              f"{e['activity_score_normiert']:>9} {e['voice_change_score']:>9} {e['aufmerksamkeits_score']:>9}")

    print(trennlinie)


def zeige_statistik(ergebnisse):
    if not ergebnisse:
        return

    def stat(feld, hinweis=""):
        werte = [e[feld] for e in ergebnisse]
        print(f"{feld:<26} Min: {min(werte):>8}  Max: {max(werte):>8}  Mittel: {round(sum(werte) / len(werte), 2):>8}  {hinweis}")

    print("\n---------- KORPUS-STATISTIK ----------")
    print(f"Anzahl Dateien: {len(ergebnisse)}\n")
    stat("events_per_minute", "<- Kandidat für MAX_EPM-Kalibrierung")
    stat("voice_changes_per_minute", "<- Kandidat für MAX_VCPM-Kalibrierung")
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


if __name__ == "__main__":
    start = time.time()
    daten = sammle_daten()
    zeige_tabelle(daten)
    zeige_statistik(daten)
    speichere_csv(daten)
    print(f"Fertig in {round(time.time() - start, 2)} Sekunden!")