import json
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

ONSET_JSON = "analysis_output/onset.json"
PITCH_JSON = "analysis_output/pitch.json"

BORDER_LOW = 180.0
BORDER_HIGH = 200.0
CENTER_HZ = 190.0


def process_pitch_to_smooth_curve(f0, max_gap_frames=4, smooth_window=5):
    f0_curve = np.copy(f0)
    n = len(f0_curve)
    is_nan = np.isnan(f0_curve)

    i = 0
    while i < n:
        if is_nan[i]:
            start = i
            while i < n and is_nan[i]:
                i += 1
            gap_size = i - start
            if gap_size <= max_gap_frames and start > 0 and i < n:
                f0_curve[start:i] = np.linspace(f0_curve[start - 1], f0_curve[i], gap_size + 2)[1:-1]
        else:
            i += 1

    is_valid = ~np.isnan(f0_curve)
    if smooth_window > 1 and np.sum(is_valid) > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        valid_indices = np.where(is_valid)[0]
        interp_full = np.interp(np.arange(n), valid_indices, f0_curve[valid_indices])
        f0_curve[is_valid] = np.convolve(interp_full, kernel, mode='same')[is_valid]

    return f0_curve


def load_and_process_data():
    with open(ONSET_JSON, "r") as f:
        onset_data = json.load(f)

    with open(PITCH_JSON, "r") as f:
        pitch_data = json.load(f)

    t_pitch = np.array([p["time_seconds"] for p in pitch_data["pitch"]])
    f0_raw = np.array([p["pitch_hz"] if p["pitch_hz"] is not None else np.nan for p in pitch_data["pitch"]])
    rms_db = np.array([p["rms_db"] for p in pitch_data["pitch"]])

    t_onsets = np.array([o["time_seconds"] for o in onset_data["onsets"]])
    t_onset_env = np.array([e["time_seconds"] for e in onset_data["envelope"]])
    onset_env = np.array([e["strength"] for e in onset_data["envelope"]])

    f0_smooth = process_pitch_to_smooth_curve(f0_raw)

    rms_norm = rms_db - np.min(rms_db)
    rms_norm = np.sqrt(rms_norm / (np.max(rms_norm) + 1e-9))

    return (t_pitch, f0_smooth, rms_norm, t_onsets, t_onset_env,
            onset_env / (np.max(onset_env) + 1e-9),
            np.interp(t_onset_env, t_pitch, rms_norm))


def plot_visual():
    (t_pitch, f0_hz, rms_norm, t_onsets, t_onset_env, onset_norm, rms_norm_onset) = load_and_process_data()

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(21, 5, figure=fig, hspace=0.2, wspace=0.05, width_ratios=[1, 1, 1, 1, 0.15])

    cmap = cm.plasma
    norm_c = mcolors.Normalize(vmin=0.6, vmax=1)

    # PLOT 1: PITCH
    ax1 = fig.add_subplot(gs[0:10, 0:4])
    ax1.axhspan(BORDER_LOW, BORDER_HIGH, color="gray", alpha=0.3, zorder=1, label="Grauzone (180–200 Hz)")
    ax1.axhline(CENTER_HZ, color="black", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2, label="Grauzone-Mitte (190 Hz)")

    for i in range(len(t_pitch) - 1):
        if not np.isnan(f0_hz[i]) and not np.isnan(f0_hz[i + 1]):
            ax1.plot(t_pitch[i:i + 2], f0_hz[i:i + 2], color=cmap(norm_c(rms_norm[i])), linewidth=3, zorder=3)

    event_pitch = np.interp(t_onsets, t_pitch, np.nan_to_num(f0_hz, nan=CENTER_HZ))
    ax1.scatter(t_onsets, event_pitch, s=100, facecolors='none', edgecolors='black', linewidths=1.5, alpha=0.6, zorder=5, label="Ereignisse / Soundeffekte")

    ax1.set_title("Pitch-Verlauf in Hz (fließende Kurve, Grauzone 180–200 Hz)", fontsize=14)
    ax1.set_ylabel("Grundfrequenz (Hz)", fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")
    ax1.set_ylim(50, 950)

    # PLOT 2: ONSET
    ax2 = fig.add_subplot(gs[11:21, 0:4])
    for i in range(len(t_onset_env) - 1):
        ax2.plot(t_onset_env[i:i + 2], onset_norm[i:i + 2], color=cmap(norm_c(rms_norm_onset[i])), linewidth=2, alpha=0.9)

    ax2.scatter(t_onsets, np.full(len(t_onsets), 0.8), color="black", s=40, zorder=5, label="Ereignispunkte")

    ax2.set_title("Ereignisanalyse: Onset Strength (lautstärke-gefärbt)", fontsize=14)
    ax2.set_ylabel("Ereignisintensität", fontsize=12)
    ax2.set_xlabel("Zeit (Sekunden)", fontsize=12)
    ax2.set_yticks([])
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right")

    # COLORBAR
    cbar_ax = fig.add_subplot(gs[:, 4])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm_c)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, pad=0.02).set_label("Normierte Lautstärke (dB)", rotation=270, labelpad=18, fontsize=12)

    plt.show()


if __name__ == "__main__":
    plot_visual()