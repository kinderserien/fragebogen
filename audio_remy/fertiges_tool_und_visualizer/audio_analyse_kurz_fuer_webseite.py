import io
import numpy as np
import librosa
import soundfile as sf

# Konfiguration
BORDER_LOW = 180.0
BORDER_HIGH = 200.0


def load_audio(file_input, max_duration=20):
    """
    Lädt Audiodaten flexibel: 
    - file_input kann ein Dateipfad (str) ODER Datei-Bytes (bytes / io.BytesIO) sein.
    """
    try:
        # Versuch 1: Fast I/O mit SoundFile
        if isinstance(file_input, bytes):
            buffer = io.BytesIO(file_input)
            y, sr = sf.read(buffer, dtype='float32')
        else:
            y, sr = sf.read(file_input, dtype='float32')

        if y.ndim > 1:
            y = np.mean(y, axis=1)  # Mono-Mischung
    except Exception:
        # Fallback 2: Librosa (für MP3s ohne Standard-Header)
        if isinstance(file_input, bytes):
            file_input = io.BytesIO(file_input)
        y, sr = librosa.load(file_input, sr=None, mono=True)

    # Audiodauer begrenzen
    if max_duration:
        max_samples = int(sr * max_duration)
        y = y[:max_samples]

    return y, sr


def analyze_audio(file_input, max_duration=20, fmin=80, fmax=900, hop_length=512):
    """
    Führt die komplette Analyse und Auswertung in einem Schritt im RAM aus.
    Gibt ein Python-Dictionary zurück, das direkt als JSON gesendet werden kann.
    """
    # 1. Audio laden
    y, sr = load_audio(file_input, max_duration=max_duration)
    
    if len(y) == 0:
        raise ValueError("Audiodatei ist leer oder konnte nicht verarbeitet werden.")

    # 2. Merkmale extrahieren (Pitch, RMS, Onsets)
    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
    t_pitch = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    t_onsets = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_length)

    # 3. Auswertung: Onset & Aktivität
    count = len(t_onsets)
    duration = float(t_onsets[-1]) if count > 0 else (len(y) / sr)
    events_per_min = round(count / (duration / 60), 2) if duration > 0 else 0.0

    rms_at_onsets = np.interp(t_onsets, t_pitch, rms_db) if count > 0 else np.array([])
    rms_mean = round(float(np.mean(rms_at_onsets)), 4) if len(rms_at_onsets) > 0 else 0.0
    rms_max = round(float(np.max(rms_at_onsets)), 4) if len(rms_at_onsets) > 0 else 0.0
    
    rms_linear = 10 ** (rms_mean / 20)
    activity_score = round(events_per_min * rms_linear, 2)

    # 4. Auswertung: Pitch-Verteilung (Vektorisiert via NumPy)
    total_samples = len(f0)
    stumm = int(np.isnan(f0).sum())
    voiced = total_samples - stumm

    if voiced > 0:
        unter_pct = round(float(np.count_nonzero(f0 < BORDER_LOW) / voiced * 100), 2)
        ueber_pct = round(float(np.count_nonzero(f0 > BORDER_HIGH) / voiced * 100), 2)
        grenze_pct = round(float(np.count_nonzero((f0 >= BORDER_LOW) & (f0 <= BORDER_HIGH)) / voiced * 100), 2)
    else:
        unter_pct = ueber_pct = grenze_pct = 0.0

    # 5. Strukturierte Ergebnisse für das Backend / die Web-API
    return {
        "duration_seconds": round(duration, 2),
        "activity": {
            "activity_score": activity_score,
            "events_per_minute": events_per_min,
            "onset_count": count,
            "rms_mean_db": rms_mean,
            "rms_max_db": rms_max
        },
        "pitch_distribution": {
            "total_samples": total_samples,
            "voiced_samples": voiced,
            "silent_samples": stumm,
            "unter_180hz_pct": unter_pct,
            "ueber_200hz_pct": ueber_pct,
            "grauzone_180_200hz_pct": grenze_pct
        }
    }


# Lokaler Test (Konsole)
if __name__ == "__main__":
    test_file = "music_source/peppa.mp3"
    result = analyze_audio(test_file)
    
    print("\n-----------------------------")
    print(f"Activity Score: {result['activity']['activity_score']}")
    print(f"Soundeffekte pro Minute: {result['activity']['events_per_minute']}\n")
    print(f"Unter 180Hz/Männerstimme: {result['pitch_distribution']['unter_180hz_pct']}%")
    print(f"Über 200Hz/Frauenstimme/Kinderstimme: {result['pitch_distribution']['ueber_200hz_pct']}%")
    print(f"Grauzone 180-200Hz: {result['pitch_distribution']['grauzone_180_200hz_pct']}%")
    print("-----------------------------\n")
