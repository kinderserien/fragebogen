import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

# ============================================================
# ===================== DATEI-EINSTELLUNGEN ===================
# ============================================================

ONSET_JSON = "analysis_output/onset.json"
PITCH_JSON = "analysis_output/pitch.json"

# Feste Grenzen aus Auswertung_neu.py
BORDER_LOW = 155.0  # Untere Grenze der Grauzone in Hz
BORDER_HIGH = 165.0  # Obere Grenze der Grauzone in Hz
CENTER_HZ = 160.0  # Mitte der Grauzone in Hz (Referenzlinie)


# ============================================================
# ===================== CURVE SMOOTHING ======================
# ============================================================

def process_pitch_to_smooth_curve(f0, max_gap_frames=4, smooth_window=5):
    """
    Bereinigt die rohen Hz-Werte für eine fließende Kurvendarstellung:
    1. Überbrückt kurze Konsonanten-Lücken.
    2. Glättet Mikrozittern der Frequenzanalyse.
    3. Belässt Stille als NaN (verhindert Abstürze auf 0 oder den Mittelwert).
    """
    f0_curve = np.copy(f0)
    n = len(f0_curve)
    is_nan = np.isnan(f0_curve)

    # 1. Kurze Mikropausen (<= max_gap_frames) linear überbrücken
    i = 0
    while i < n:
        if is_nan[i]:
            start = i
            while i < n and is_nan[i]:
                i += 1
            end = i
            gap_size = end - start
            if gap_size <= max_gap_frames and start > 0 and end < n:
                val_start = f0_curve[start - 1]
                val_end = f0_curve[end]
                f0_curve[start:end] = np.linspace(val_start, val_end, gap_size + 2)[1:-1]
        else:
            i += 1

    # 2. Gleitende Glättung (Filter) anwenden
    is_valid = ~np.isnan(f0_curve)
    if smooth_window > 1 and np.sum(is_valid) > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        valid_indices = np.where(is_valid)[0]

        interp_full = np.interp(np.arange(n), valid_indices, f0_curve[valid_indices])
        smoothed_full = np.convolve(interp_full, kernel, mode='same')

        f0_curve[is_valid] = smoothed_full[is_valid]

    return f0_curve


# ============================================================
# ===================== DATEN VERARBEITEN ====================
# ============================================================

def load_and_process_data():
    with open(ONSET_JSON, "r") as f:
        onset_data = json.load(f)

    with open(PITCH_JSON, "r") as f:
        pitch_data = json.load(f)

    # --- Pitch & RMS extrahieren ---
    t_pitch = np.array([p["time_seconds"] for p in pitch_data["pitch"]])
    f0_raw = np.array([p["pitch_hz"] if p["pitch_hz"] is not None else np.nan for p in pitch_data["pitch"]])
    rms_db = np.array([p["rms_db"] for p in pitch_data["pitch"]])

    # --- Onsets & Envelope extrahieren ---
    t_onsets = np.array([o["time_seconds"] for o in onset_data["onsets"]])
    t_onset_env = np.array([e["time_seconds"] for e in onset_data["envelope"]])
    onset_env = np.array([e["strength"] for e in onset_data["envelope"]])

    # --- Pitch in echte, glatte Hz-Kurve umwandeln (KEINE Normierung) ---
    f0_smooth = process_pitch_to_smooth_curve(f0_raw, max_gap_frames=4, smooth_window=5)

    # --- RMS & Onset Normalisierung für Farbcodierung ---
    rms_norm = (rms_db - np.min(rms_db))
    rms_norm /= (np.max(rms_norm) + 1e-9)
    rms_norm = np.sqrt(rms_norm)

    onset_norm = onset_env / (np.max(onset_env) + 1e-9)
    rms_norm_onset = np.interp(t_onset_env, t_pitch, rms_norm)

    return (t_pitch, f0_smooth, rms_norm,
            t_onsets, t_onset_env, onset_norm, rms_norm_onset)


# ============================================================
# ===================== VISUALISIERUNG ========================
# ============================================================

def plot_visual():
    (t_pitch, f0_hz, rms_norm,
     t_onsets, t_onset_env, onset_norm, rms_norm_onset) = load_and_process_data()

    # Figure & GridSpec Layout
    fig = plt.figure(figsize=(20, 12))

    gs = gridspec.GridSpec(
        21, 5,
        figure=fig,
        hspace=0.2,
        wspace=0.05,
        width_ratios=[1, 1, 1, 1, 0.15]
    )

    cmap = cm.plasma
    norm_c = mcolors.Normalize(vmin=0.6, vmax=1)

    # ============================================================
    # ===================== PLOT 1: PITCH (Hz) ===================
    # ============================================================

    ax1 = fig.add_subplot(gs[0:10, 0:4])

    # Grauzone als farblicher Korridor hinterlegen (180–200 Hz)
    ax1.axhspan(
        BORDER_LOW, BORDER_HIGH,
        color="gray",
        alpha=0.3,
        zorder=1,
        label="Grauzone (180–200 Hz)"
    )

    # Referenzlinie bei 190 Hz (Mitte der Grauzone)
    ax1.axhline(
        CENTER_HZ,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        zorder=2,
        label="Grauzone-Mitte (190 Hz)"
    )

    # Frequenzverlauf in ECHTEN Hertz zeichnen (ohne Z-Score/Normierung)
    for i in range(len(t_pitch) - 1):
        if not np.isnan(f0_hz[i]) and not np.isnan(f0_hz[i + 1]):
            ax1.plot(
                t_pitch[i:i + 2],
                f0_hz[i:i + 2],
                color=cmap(norm_c(rms_norm[i])),
                linewidth=3,
                zorder=3
            )

    # Pitch-Werte an den Ereigniszeiten abgreifen
    event_pitch = np.interp(t_onsets, t_pitch, np.nan_to_num(f0_hz, nan=CENTER_HZ))

    ax1.scatter(
        t_onsets,
        event_pitch,
        s=100,
        facecolors='none',
        edgecolors='black',
        linewidths=1.5,
        alpha=0.6,
        zorder=5,
        label="Ereignisse / Soundeffekte",
        marker='o'
    )

    ax1.set_title("Pitch-Verlauf in Hz (fließende Kurve, Grauzone 180–200 Hz)", fontsize=14)
    ax1.set_ylabel("Grundfrequenz (Hz)", fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    # Automatischer Y-Fokus, damit extreme Ausreißer nach oben das Diagramm nicht zu stark stauchen
    # Y-Achse fest auf 50 Hz bis 1000 Hz setzen
    ax1.set_ylim(50, 950)

    # ============================================================
    # ===================== PLOT 2: ONSET =========================
    # ============================================================

    ax2 = fig.add_subplot(gs[11:21, 0:4])

    for i in range(len(t_onset_env) - 1):
        ax2.plot(
            t_onset_env[i:i + 2],
            onset_norm[i:i + 2],
            color=cmap(norm_c(rms_norm_onset[i])),
            linewidth=2,
            alpha=0.9
        )

    ax2.scatter(
        t_onsets,
        0.8 * np.ones(len(t_onsets)),
        color="black",
        s=40,
        zorder=5,
        label="Ereignispunkte",
        marker='o',
        alpha=1.0
    )

    ax2.set_title("Ereignisanalyse: Onset Strength (lautstärke-gefärbt)", fontsize=14)
    ax2.set_ylabel("Ereignisintensität", fontsize=12)
    ax2.set_xlabel("Zeit (Sekunden)", fontsize=12)
    ax2.set_yticks([])
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right")

    # ============================================================
    # ===================== COLORBAR ==============================
    # ============================================================

    cbar_ax = fig.add_subplot(gs[:, 4])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm_c)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cbar_ax, pad=0.02)
    cbar.set_label("Normierte Lautstärke (dB)", rotation=270, labelpad=18, fontsize=12)

    plt.show()


if __name__ == "__main__":
    plot_visual()
