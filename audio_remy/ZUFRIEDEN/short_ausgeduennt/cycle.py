import os
from analyse_main_24_07_26 import Config, analyze_file

KORPUS_ORDNER = "korpus"
SUPPORTED_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")


def durchlaufe_korpus(ordner=KORPUS_ORDNER):
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
            ergebnisse.append({
                "titel": dateiname,
                "stimme_valenz": res["stimme_valenz"],
                "activity_score_normiert": res["activity_score_normiert"],
                "voice_change_score": res["voice_change_score"],
                "aufmerksamkeits_score": res["aufmerksamkeits_score"],
            })
        except Exception as e:
            print(f"  [FEHLER] {dateiname} konnte nicht analysiert werden: {e}")

    return ergebnisse


def zeige_tabelle(ergebnisse):
    if not ergebnisse:
        print("Keine Daten zum Anzeigen.")
        return

    header = f"{'Titel':<35} {'Stimme-V':>9} {'ACS':>9} {'VCS':>9} {'GESAMT':>9}"
    trennlinie = "-" * len(header)

    print("\n" + trennlinie)
    print(header)
    print(trennlinie)

    for e in ergebnisse:
        print(f"{e['titel']:<35} {e['stimme_valenz']:>9} {e['activity_score_normiert']:>9} "
              f"{e['voice_change_score']:>9} {e['aufmerksamkeits_score']:>9}")

    print(trennlinie + "\n")


if __name__ == "__main__":
    zeige_tabelle(durchlaufe_korpus())