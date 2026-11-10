"""Phase-independent swing and support metrics from measured contacts."""
import numpy as np


def measured_swing(contact, clearance, dt=.02):
    counts, peaks, times, low, low_samples, sample_counts, airborne = [], [], [], [], [], [], []
    for touching, height in zip(contact.T, clearance.T):
        departures = np.flatnonzero(touching[:-1] & ~touching[1:]) + 1
        returns = np.flatnonzero(~touching[:-1] & touching[1:]) + 1
        swings = []
        for start in departures:
            i = np.searchsorted(returns, start, side="right")
            if i < len(returns) and returns[i] - start >= 2:
                swings.append(height[start:returns[i]])
        swing_peaks = np.asarray([float(s.max()) for s in swings])
        counts.append(len(swings))
        peaks.append(float(np.median(swing_peaks)) if swings else None)
        times.append(float(dt * np.median([len(s) for s in swings])) if swings else None)
        low.append(float((swing_peaks < .01).mean()) if swings else None)
        sample_counts.append(sum(map(len, swings)))
        low_samples.append(float((np.concatenate(swings) < .01).mean()) if swings else None)
        airborne.append(float((~touching).mean()))
    return dict(swing_count=counts, swing_peak_median_m=peaks, swing_duration_median_s=times,
                swing_peak_below_1cm_fraction=low, swing_samples_below_1cm_fraction=low_samples,
                swing_sample_count=sample_counts,
                airborne_fraction=airborne, feet_without_completed_swing=[i for i, n in enumerate(counts) if n == 0])
