import os

from analyse_main_23_07_26 import Config, analyze_file

# -------------KONFIGURATION-------------
KORPUS_ORDNER = "korpus"      # Ordner mit den zu analysierenden Audiodateien
SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")


def durchlaufe_korpus(ordner=KORPUS_ORDNER):
    """
    Durchläuft alle unterstützten Audiodateien in 'ordner' und berechnet
    für jede die drei Aufmerksamkeits-Scores (Stimme, Activity, Gesamt).
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
                "stimme_score":             res["stimme_score"],
                "activity_score_normiert":  res["activity_score_normiert"],
                "gesamt_score":             res["gesamt_score"],
            })
        except Exception as e:
            print(f"  [FEHLER] {dateiname} konnte nicht analysiert werden: {e}")

    return ergebnisse


def zeige_tabelle(ergebnisse):
    if not ergebnisse:
        print("Keine Daten zum Anzeigen.")
        return

    header = f"{'Titel':<35} {'Stimme':>9} {'Activity':>10} {'GESAMT':>9}"
    trennlinie = "-" * len(header)

    print("\n" + trennlinie)
    print(header)
    print(trennlinie)

    for e in ergebnisse:
        print(f"{e['titel']:<35} {e['stimme_score']:>9} "
              f"{e['activity_score_normiert']:>10} {e['gesamt_score']:>9}")

    print(trennlinie + "\n")


def main():
    ergebnisse = durchlaufe_korpus()
    zeige_tabelle(ergebnisse)


if __name__ == "__main__":
    main()