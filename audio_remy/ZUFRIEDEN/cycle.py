import os

from Main_Analyse_27_07_26.py import Config, analyze_file

# -------------KONFIGURATION-------------
KORPUS_ORDNER = "korpus"      # Ordner mit den zu analysierenden Audiodateien
SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")


def durchlaufe_korpus(ordner=KORPUS_ORDNER):
    """
    Durchläuft alle unterstützten Audiodateien in 'ordner' und berechnet
    für jede die aktuellen Aufmerksamkeits-Scores aus
    analyse_main_24_07_26 / analyse_auswertung_24_07_26:
    Stimme-Valenz, Activity-Score normiert (ACS), Stimmwechsel-Score (VCS)
    und den daraus zusammengesetzten Gesamt-Aufmerksamkeitsscore.
    """
    if not os.path.isdir(ordner):
        print(f"[FEHLER] Ordner nicht gefunden: {ordner}")
        return []

    config = Config()  # gleiche Analyse-Parameter für alle Dateien

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

    header = (f"{'Titel':<35} {'Stimme-V':>9} {'ACS':>9} "
              f"{'VCS':>9} {'GESAMT':>9}")
    trennlinie = "-" * len(header)

    print("\n" + trennlinie)
    print(header)
    print(trennlinie)

    for e in ergebnisse:
        print(f"{e['titel']:<35} {e['stimme_valenz']:>9} {e['activity_score_normiert']:>9} "
              f"{e['voice_change_score']:>9} {e['aufmerksamkeits_score']:>9}")

    print(trennlinie + "\n")


def main():
    ergebnisse = durchlaufe_korpus()
    zeige_tabelle(ergebnisse)


if __name__ == "__main__":
    main()
