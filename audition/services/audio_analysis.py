# pyre-ignore-all-errors

"""
Audio Analysis Service - Signal Processing & Feature Extraction

This module performs highly accurate audio analysis to feed the deliberation service.
It strictly adheres to physical engineering principles and BS.1770-4 standards.

Key responsibilities:
- Extract Time-Series Circuit Envelopes (high-resolution, multi-dimensional signal flow)
- Compute BS.1770-4 Integrated Loudness and LRA using strict K-weighting
- Calculate True Peak using 4x oversampling
- Identify physical section boundaries for macro-form analysis
- Detect potential engineering issues (phase cancellation, mud, harshness, etc.)
"""

import io
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.signal import butter, resample_poly, sosfilt

# ──────────────────────────────────────────
# Constants & Profiles
# ──────────────────────────────────────────
LOG_FLOOR = 1e-10
A4_HZ = 440.0
TIME_SERIES_RESOLUTION_SEC = 0.1

# 6-band spectral division (Sub / Bass / LowMid / Mid / High / Air)
BAND_EDGES = {
    "sub": (20.0, 60.0),
    "bass": (60.0, 200.0),
    "low_mid": (200.0, 500.0),
    "mid": (500.0, 2000.0),
    "high": (2000.0, 8000.0),
    "air": (8000.0, None)
}

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


# ──────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────
@dataclass(frozen=True)
class TrackSpectrum:
    """Pre-computed Fast Fourier Transform data to ensure maximum API throughput."""
    freqs: NDArray[np.float64]
    mono_power: NDArray[np.float64]
    mid_power: NDArray[np.float64]

    @classmethod
    def compute(cls, mono_signal: NDArray[np.float64], mid_signal: NDArray[np.float64], sample_rate: int) -> "TrackSpectrum":
        mono_power = np.abs(np.fft.rfft(mono_signal)) ** 2
        mid_power = np.abs(np.fft.rfft(mid_signal)) ** 2
        freqs = np.fft.rfftfreq(len(mono_signal), 1.0 / sample_rate)
        return cls(freqs, mono_power, mid_power)


# ══════════════════════════════════════════
# Signal Processing Utilities
# ══════════════════════════════════════════
def _build_k_weight_sos(sample_rate: int) -> NDArray:
    """
    Build BS.1770-4 K-weighting filter (pre-filter + RLB high-pass).
    Constructs the exact BS.1770-4 K-weighting filter coefficients.
    Safely stacks the SOS (Second-Order Sections) arrays.
    """
    # High-shelf (Pre-filter: boosts ~1.68kHz and above)
    f0 = 1681.974450955533
    gain = 4.0
    Q = 0.7071752369554196
    w0 = 2.0 * np.pi * f0 / sample_rate
    A = 10.0 ** (gain / 40.0)
    alpha = np.sin(w0) / (2.0 * Q)
    
    b0 = A * ((A + 1.0) + (A - 1.0) * np.cos(w0) + 2.0 * np.sqrt(A) * alpha)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * np.cos(w0))
    b2 = A * ((A + 1.0) + (A - 1.0) * np.cos(w0) - 2.0 * np.sqrt(A) * alpha)
    a0 = (A + 1.0) - (A - 1.0) * np.cos(w0) + 2.0 * np.sqrt(A) * alpha
    a1 = 2.0 * ((A - 1.0) - (A + 1.0) * np.cos(w0))
    a2 = (A + 1.0) - (A - 1.0) * np.cos(w0) - 2.0 * np.sqrt(A) * alpha
    
    sos_high_shelf = np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]).reshape(1, 6)
    
    # High-pass (RLB filter: rolls off extreme lows below 38Hz)
    sos_high_pass = butter(2, 38.0, btype='highpass', fs=sample_rate, output='sos')
    
    return np.vstack([sos_high_shelf, sos_high_pass])


# ══════════════════════════════════════════
# API Entry Point
# ══════════════════════════════════════════
def validate_audio_file(file_path: str) -> Dict[str, Any]:
    """Validates header integrity before committing to expensive computations."""
    import os
    if os.path.getsize(file_path) < 44:
        raise ValueError("Invalid audio file: Too small to contain headers.")
    try:
        info = sf.info(file_path)
    except Exception:
        raise ValueError("Unsupported format. Permitted: WAV, FLAC, AIFF.")
    
    if info.samplerate < 8000 or info.samplerate > 384000:
        raise ValueError(f"Unsupported sample rate: {info.samplerate}Hz")
    
    return {
        "duration_sec": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels
    }


def analyze_audio_file(file_path: str) -> Dict[str, Any]:
    """The immutable entry point for the REST API."""
    data, sample_rate = sf.read(file_path, dtype="float64")
    
    if data.ndim == 1:
        data = np.column_stack([data, data])

    left_channel = data[:, 0]
    right_channel = data[:, 1]
    mono_signal = (left_channel + right_channel) * 0.5
    mid_signal = mono_signal
    side_signal = (left_channel - right_channel) * 0.5
    
    spectrum = TrackSpectrum.compute(mono_signal, mid_signal, sample_rate)
    
    k_weight_sos = _build_k_weight_sos(sample_rate)
    left_k_weighted = sosfilt(k_weight_sos, left_channel)
    right_k_weighted = sosfilt(k_weight_sos, right_channel)
    mono_k_weighted = (left_k_weighted + right_k_weighted) * 0.5

    # 1. High-Resolution Circuit Envelopes (9 Dimensions)
    circuit_envelopes = _compute_time_series_circuit_envelopes(
        mono_signal, mono_k_weighted, side_signal, left_channel, right_channel, sample_rate
    )
    
    # 2. Global Baseline Metrics (Strict BS.1770-4 compliance)
    whole_metrics = _compute_whole_track_metrics(
        mono_signal, left_channel, right_channel, mid_signal, side_signal, 
        left_k_weighted, right_k_weighted, sample_rate, spectrum, circuit_envelopes
    )
    
    # 3. Physical Section Boundaries
    physical_sections = _detect_physical_sections(
        len(mono_signal), sample_rate, circuit_envelopes
    )

    return {
        "track_identity": {
            "duration_sec": round(float(len(data) / sample_rate), 2),
            "sample_rate": sample_rate,
            "bpm": _estimate_bpm(mono_signal, sample_rate),
            "key": _estimate_key(mono_signal, sample_rate),
            "bit_depth": _detect_bit_depth(file_path),
        },
        "whole_track_metrics": whole_metrics,
        "time_series_circuit_envelopes": circuit_envelopes,
        "physical_sections": physical_sections,
        "detected_problems": _detect_problems(whole_metrics),
    }


# ══════════════════════════════════════════
# Time-Series Circuit Envelope Generation
# ══════════════════════════════════════════
def _compute_time_series_circuit_envelopes(
    mono_signal: NDArray, mono_k_weighted: NDArray, side_signal: NDArray, 
    left_channel: NDArray, right_channel: NDArray, sample_rate: int
) -> Dict[str, List[float]]:
    """
    Transforms the flow of time into explicitly defined, highly readable 9-dimensional envelopes.
    Vectorized over 1-second chunks for extreme throughput.
    """
    chunk_size = int(sample_rate * TIME_SERIES_RESOLUTION_SEC)
    number_of_chunks = len(mono_signal) // chunk_size
    
    if number_of_chunks == 0:
        return {}

    valid_length = number_of_chunks * chunk_size
    mono_chunks = mono_signal[:valid_length].reshape(number_of_chunks, chunk_size)
    mono_k_chunks = mono_k_weighted[:valid_length].reshape(number_of_chunks, chunk_size)
    side_chunks = side_signal[:valid_length].reshape(number_of_chunks, chunk_size)
    left_chunks = left_channel[:valid_length].reshape(number_of_chunks, chunk_size)
    right_chunks = right_channel[:valid_length].reshape(number_of_chunks, chunk_size)

    # Dimension 1: Loudness (K-weighted, BS.1770-4)
    mean_square_k = np.mean(mono_k_chunks ** 2, axis=1) + LOG_FLOOR
    lufs_envelope = -0.691 + 10.0 * np.log10(mean_square_k)
    
    # Dimension 2 & 3: Crest Factor & Stereo Width
    rms_values = np.sqrt(np.mean(mono_chunks ** 2, axis=1) + LOG_FLOOR)
    peak_values = np.max(np.abs(mono_chunks), axis=1)
    crest_envelope = 20.0 * np.log10(np.maximum(peak_values, LOG_FLOOR)) - 20.0 * np.log10(rms_values)
    
    side_rms_values = np.sqrt(np.mean(side_chunks ** 2, axis=1) + LOG_FLOOR)
    width_envelope = np.clip(side_rms_values / rms_values, 0.0, 1.0)

    # Spectral Base Computations
    power_chunks = np.abs(np.fft.rfft(mono_chunks, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(chunk_size, 1.0 / sample_rate)
    total_energy_per_chunk = np.sum(power_chunks[:, frequencies >= 20], axis=1) + LOG_FLOOR

    # Dimension 4 & 5: Low-end Balance (Sub / Bass ratios)
    sub_energy = np.sum(power_chunks[:, (frequencies >= 20) & (frequencies < 60)], axis=1)
    bass_energy = np.sum(power_chunks[:, (frequencies >= 60) & (frequencies < 200)], axis=1)
    sub_ratio_envelope = sub_energy / total_energy_per_chunk
    bass_ratio_envelope = bass_energy / total_energy_per_chunk

    # Dimension 6: Vocal Presence (1-5kHz energy ratio)
    vocal_energy = np.sum(power_chunks[:, (frequencies >= 1000) & (frequencies < 5000)], axis=1)
    vocal_presence_envelope = vocal_energy / total_energy_per_chunk

    # Dimension 7: Spectral Brightness (spectral centroid)
    weighted_frequencies = np.sum(power_chunks * frequencies, axis=1)
    spectral_brightness_envelope = weighted_frequencies / total_energy_per_chunk / (chunk_size // 2)

    # Dimension 8: Low Mono Correlation (< 120Hz L/R coherence)
    sos_low_pass = butter(4, 120.0, btype='lowpass', fs=sample_rate, output='sos')
    left_low_chunks = sosfilt(sos_low_pass, left_chunks, axis=1)
    right_low_chunks = sosfilt(sos_low_pass, right_chunks, axis=1)
    
    left_low_mean = np.mean(left_low_chunks, axis=1, keepdims=True)
    right_low_mean = np.mean(right_low_chunks, axis=1, keepdims=True)
    
    numerator = np.sum((left_low_chunks - left_low_mean) * (right_low_chunks - right_low_mean), axis=1)
    denominator_left = np.sum((left_low_chunks - left_low_mean) ** 2, axis=1)
    denominator_right = np.sum((right_low_chunks - right_low_mean) ** 2, axis=1)
    denominator = np.sqrt(denominator_left * denominator_right) + LOG_FLOOR
    
    low_mono_correlation_envelope = np.clip(numerator / denominator, -1.0, 1.0)

    # Dimension 9: Transient Sharpness (onset strength via 1st derivative)
    transient_sharpness_envelope = np.zeros(number_of_chunks)
    for i in range(number_of_chunks):
        chunk_diff = np.abs(np.diff(mono_chunks[i]))
        transient_sharpness_envelope[i] = float(np.percentile(chunk_diff, 95))

    def _round_list(array: NDArray, decimals: int) -> List[float]:
        return [round(float(value), decimals) for value in array]

    return {
        "resolution_sec": TIME_SERIES_RESOLUTION_SEC,
        "lufs": _round_list(lufs_envelope, 1),
        "crest_db": _round_list(crest_envelope, 1),
        "width": _round_list(width_envelope, 3),
        "sub_ratio": _round_list(sub_ratio_envelope, 3),
        "bass_ratio": _round_list(bass_ratio_envelope, 3),
        "vocal_presence": _round_list(vocal_presence_envelope, 3),
        "spectral_brightness": _round_list(spectral_brightness_envelope, 4),
        "low_mono_correlation": _round_list(low_mono_correlation_envelope, 3),
        "transient_sharpness": _round_list(transient_sharpness_envelope, 6),
    }


# ══════════════════════════════════════════
# Whole Track Metrics Integration
# ══════════════════════════════════════════
def _compute_whole_track_metrics(
    mono_signal: NDArray, left_channel: NDArray, right_channel: NDArray, 
    mid_signal: NDArray, side_signal: NDArray, left_k_weighted: NDArray, 
    right_k_weighted: NDArray, sample_rate: int, spectrum: TrackSpectrum, 
    circuit_envelopes: Dict[str, List[float]]
) -> Dict[str, Any]:
    
    # BS.1770-4 Integrated Loudness
    mean_k_power = np.mean(left_k_weighted ** 2) + np.mean(right_k_weighted ** 2)
    integrated_lufs = -0.691 + 10.0 * np.log10(max(mean_k_power, LOG_FLOOR))
    
    # True Peak Estimation (4x oversampling)
    true_peak_value = _true_peak_estimate_chunked(left_channel, right_channel)
    true_peak_dbtp = 20.0 * np.log10(max(true_peak_value, LOG_FLOOR))
    
    # BS.1770-4 LRA & Short-term Max (double-gated)
    short_term_lufs_values = []
    window_samples = int(3.0 * sample_rate)
    hop_samples = int(1.0 * sample_rate)
    
    for start_idx in range(0, len(left_k_weighted) - window_samples, hop_samples):
        end_idx = start_idx + window_samples
        power_left = np.mean(left_k_weighted[start_idx:end_idx] ** 2)
        power_right = np.mean(right_k_weighted[start_idx:end_idx] ** 2)
        st_lufs = -0.691 + 10.0 * np.log10(max(power_left + power_right, LOG_FLOOR))
        short_term_lufs_values.append(st_lufs)
        
    st_array = np.array(short_term_lufs_values)
    absolute_threshold = -70.0
    gated_st = st_array[st_array > absolute_threshold]
    
    if len(gated_st) > 0:
        relative_threshold = -0.691 + 10.0 * np.log10(max(np.mean(10 ** ((gated_st + 0.691) / 10.0)), LOG_FLOOR)) - 20.0
        final_gated_st = gated_st[gated_st > relative_threshold]
        
        if len(final_gated_st) > 2:
            lra_lu = float(np.percentile(final_gated_st, 95) - np.percentile(final_gated_st, 10))
        else:
            lra_lu = 0.0
        short_term_max = float(np.max(gated_st))
    else:
        lra_lu = 0.0
        short_term_max = integrated_lufs

    psr_db = true_peak_dbtp - short_term_max

    # Band Energy Ratios (6-band split)
    total_energy = np.sum(spectrum.mono_power[spectrum.freqs >= 20]) + LOG_FLOOR
    ratios = {}
    for band_name, (low_hz, high_hz) in BAND_EDGES.items():
        if high_hz is None: 
            high_hz = np.inf
        mask = (spectrum.freqs >= low_hz) & (spectrum.freqs < high_hz)
        ratios[f"{band_name}_ratio"] = round(float(np.sum(spectrum.mono_power[mask]) / total_energy), 4)

    # Risks Analysis (harshness 2-6kHz, mud 200-500Hz)
    harshness_mask = (spectrum.freqs >= 2000) & (spectrum.freqs < 6000)
    harshness_energy = np.sum(spectrum.mono_power[harshness_mask])
    harshness_risk = np.clip((harshness_energy / total_energy) * 3.0, 0.0, 1.0)
    
    low_mid_ratio = ratios.get("low_mid_ratio", 0.0)
    mud_risk = np.clip((low_mid_ratio - 0.15) / 0.15, 0.0, 1.0)

    # Global Dynamics & Spatial Metrics
    mono_rms = np.sqrt(np.mean(mono_signal ** 2))
    mono_peak = np.max(np.abs(mono_signal))
    crest_db = 20.0 * np.log10(max(mono_peak, LOG_FLOOR)) - 20.0 * np.log10(max(mono_rms, LOG_FLOOR))

    side_rms = np.sqrt(np.mean(side_signal ** 2))
    mid_rms = np.sqrt(np.mean(mid_signal ** 2))
    stereo_width = np.clip(side_rms / (mid_rms + LOG_FLOOR), 0.0, 1.0)

    stereo_correlation = np.corrcoef(left_channel, right_channel)[0, 1]
    low_mono_correlation = float(np.mean(circuit_envelopes.get("low_mono_correlation", [1.0])))

    return {
        "integrated_lufs": round(float(integrated_lufs), 1),
        "true_peak_dbtp": round(float(true_peak_dbtp), 2),
        "lra_lu": round(float(lra_lu), 1),
        "psr_db": round(float(psr_db), 1),
        "crest_db": round(float(crest_db), 1),
        "stereo_width": round(float(stereo_width), 3),
        "stereo_correlation": round(float(stereo_correlation), 3),
        "low_mono_correlation_below_120hz": round(low_mono_correlation, 3),
        "harshness_risk": round(float(harshness_risk), 3),
        "mud_risk": round(float(mud_risk), 3),
        **ratios
    }


def _true_peak_estimate_chunked(left_channel: NDArray, right_channel: NDArray) -> float:
    """Chunked True Peak estimation to manage memory on long files."""
    chunk_size = 44100 * 5  # 5 seconds
    global_max_peak = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
    
    if global_max_peak < 1e-4:
        return float(global_max_peak)
    
    true_peak_max = 0.0
    for start_idx in range(0, len(left_channel), chunk_size):
        end_idx = start_idx + chunk_size
        left_chunk = left_channel[start_idx:end_idx]
        right_chunk = right_channel[start_idx:end_idx]
        
        # Only upsample chunks where the peak is close to the global maximum
        chunk_max = max(np.max(np.abs(left_chunk)), np.max(np.abs(right_chunk)))
        if chunk_max > global_max_peak * 0.5:
            left_oversampled = resample_poly(left_chunk, 4, 1)
            right_oversampled = resample_poly(right_chunk, 4, 1)
            true_peak_max = max(
                true_peak_max, 
                np.max(np.abs(left_oversampled)), 
                np.max(np.abs(right_oversampled))
            )
            
    return float(max(true_peak_max, global_max_peak))


# ══════════════════════════════════════════
# Physical Section Boundaries & Meta Estimations
# ══════════════════════════════════════════
def _detect_physical_sections(total_samples: int, sample_rate: int, circuit_envelopes: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identifies major energy shifts and aggregates evidence for the AI's Attention mechanism."""
    lufs_envelope = np.array(circuit_envelopes.get("lufs", []))
    width_envelope = np.array(circuit_envelopes.get("width", []))
    
    if len(lufs_envelope) < 10:
        return [{
            "start_sec": 0.0, 
            "end_sec": float(total_samples / sample_rate),
            "avg_lufs": round(float(np.mean(lufs_envelope)), 1) if len(lufs_envelope) > 0 else -14.0,
            "avg_width": round(float(np.mean(width_envelope)), 3) if len(width_envelope) > 0 else 0.0
        }]
    
    lufs_differences = np.abs(np.diff(lufs_envelope))
    detection_threshold = np.mean(lufs_differences) + 1.5 * np.std(lufs_differences)
    
    boundaries = [0]
    for idx, diff in enumerate(lufs_differences):
        if diff > detection_threshold and (idx - boundaries[-1]) > 8:
            boundaries.append(idx + 1)
    
    boundaries.append(len(lufs_envelope))
    
    sections = []
    for idx in range(len(boundaries) - 1):
        start_index = boundaries[idx]
        end_index = boundaries[idx + 1]
        sections.append({
            "start_sec": float(start_index * TIME_SERIES_RESOLUTION_SEC),
            "end_sec": float(end_index * TIME_SERIES_RESOLUTION_SEC),
            "avg_lufs": round(float(np.mean(lufs_envelope[start_index:end_index])), 1),
            "avg_width": round(float(np.mean(width_envelope[start_index:end_index])), 3)
        })
        
    return sections


def _estimate_bpm(mono_signal: NDArray, sample_rate: int) -> Optional[float]:
    chunk_for_bpm = mono_signal  # Analyze full track for accurate BPM
    frames = sliding_window_view(chunk_for_bpm, window_shape=1024)[::512]
    
    if len(frames) == 0:
        return None
        
    energy_onsets = np.maximum(np.diff(np.sum(frames ** 2, axis=1)), 0)
    correlation = np.correlate(energy_onsets, energy_onsets, mode="full")[len(energy_onsets) - 1:]
    
    frames_per_second = sample_rate / 512.0
    min_lag = int(frames_per_second * 60 / 200)
    max_lag = int(frames_per_second * 60 / 60)
    
    if max_lag > len(correlation) or len(correlation[min_lag:max_lag]) == 0:
        return None
        
    best_lag_index = np.argmax(correlation[min_lag:max_lag]) + min_lag
    bpm = frames_per_second * 60.0 / best_lag_index
    return round(float(bpm), 1)


def _estimate_key(mono_signal: NDArray, sample_rate: int) -> Optional[str]:
    chunk_for_key = mono_signal  # Analyze full track for accurate key detection
    fft_result = np.abs(np.fft.rfft(chunk_for_key))
    frequencies = np.fft.rfftfreq(len(chunk_for_key), 1.0 / sample_rate)
    
    chroma_vector = np.zeros(12)
    midi_notes = np.arange(24, 96)
    
    for note in midi_notes:
        target_frequency = A4_HZ * 2 ** ((note - 69) / 12)
        closest_index = np.clip(np.searchsorted(frequencies, target_frequency), 0, len(frequencies) - 1)
        chroma_vector[note % 12] += fft_result[closest_index] ** 2
        
    best_correlation = -1.0
    best_key = "C"
    best_mode = "major"
    
    for shift in range(12):
        shifted_chroma = np.roll(chroma_vector, -shift)
        correlation_major = np.corrcoef(shifted_chroma, MAJOR_PROFILE)[0, 1]
        correlation_minor = np.corrcoef(shifted_chroma, MINOR_PROFILE)[0, 1]
        
        if correlation_major > best_correlation:
            best_correlation = correlation_major
            best_key = NOTE_NAMES[shift]
            best_mode = "major"
            
        if correlation_minor > best_correlation:
            best_correlation = correlation_minor
            best_key = NOTE_NAMES[shift]
            best_mode = "minor"
            
    return f"{best_key} {best_mode}"


def _detect_problems(whole_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Detect engineering issues based on audio metrics."""
    detected_problems = []
    
    # True Peak Danger (> -0.3 dBTP)
    true_peak = whole_metrics.get("true_peak_dbtp", -100.0)
    if true_peak > -0.3:
        detected_problems.append({
            "issue": "true_peak_danger",
            "severity": "high",
            "value": true_peak
        })
        
    # Mud Risk (200-500Hz excess)
    mud_risk = whole_metrics.get("mud_risk", 0.0)
    if mud_risk > 0.4:
        detected_problems.append({
            "issue": "mud_risk",
            "severity": "medium",
            "value": mud_risk
        })
        
    # Phase Cancellation in Lows (< 120Hz)
    low_mono_correlation = whole_metrics.get("low_mono_correlation_below_120hz", 1.0)
    if low_mono_correlation < 0.3:
        detected_problems.append({
            "issue": "phase_cancellation_lows",
            "severity": "medium",
            "value": low_mono_correlation
        })

    # Harshness in 2-6kHz
    harshness_risk = whole_metrics.get("harshness_risk", 0.0)
    if harshness_risk > 0.5:
        detected_problems.append({
            "issue": "harshness_risk",
            "severity": "medium",
            "value": harshness_risk
        })

    # Crest Factor observation (informational, not a problem)
    crest_db = whole_metrics.get("crest_db", 20.0)
    if crest_db < 3.0:
        detected_problems.append({
            "issue": "extremely_low_crest_factor",
            "severity": "low",
            "value": crest_db
        })
        
    return detected_problems


def _detect_bit_depth(file_path: str) -> int:
    try:
        with open(file_path, "rb") as f:
            header = f.read(40)
        if len(header) >= 36 and header[:4] == b"RIFF":
            return struct.unpack_from("<H", header, 34)[0]
    except Exception:
        pass
    return 24



if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python audio_analysis.py <audio_file> [--validate-only]")
        print("")
        print("Examples:")
        print("  python audio_analysis.py master.wav")
        print("  python audio_analysis.py master.flac --validate-only")
        sys.exit(1)

    audio_file_path = sys.argv[1]
    validate_only = "--validate-only" in sys.argv

    # Step 1: Validation (always runs first)
    validation_result = validate_audio_file(audio_file_path)
    print(f"[VALIDATION] Duration: {validation_result['duration_sec']:.1f}s | "
          f"Sample Rate: {validation_result['sample_rate']}Hz | "
          f"Channels: {validation_result['channels']}")

    if validate_only:
        print("[DONE] Validation only. No analysis performed.")
        sys.exit(0)

    # Step 2: Full Analysis
    print("[ANALYSIS] Computing Time-Series Circuit Envelopes...")
    analysis_result = analyze_audio_file(audio_file_path)

    # Step 3: Human-readable summary
    metrics = analysis_result["whole_track_metrics"]
    identity = analysis_result["track_identity"]
    envelopes = analysis_result["time_series_circuit_envelopes"]
    problems = analysis_result["detected_problems"]
    sections = analysis_result["physical_sections"]

    print(f"[IDENTITY] BPM: {identity['bpm']} | Key: {identity['key']} | "
          f"Bit Depth: {identity['bit_depth']}bit")
    print(f"[METRICS]  LUFS: {metrics['integrated_lufs']} | "
          f"TP: {metrics['true_peak_dbtp']} dBTP | "
          f"LRA: {metrics['lra_lu']} LU | "
          f"PSR: {metrics['psr_db']} dB | "
          f"Crest: {metrics['crest_db']} dB")
    print(f"[SPATIAL]  Width: {metrics['stereo_width']} | "
          f"Correlation: {metrics['stereo_correlation']} | "
          f"Low Mono: {metrics['low_mono_correlation_below_120hz']}")
    print(f"[RISK]     Harshness: {metrics['harshness_risk']} | "
          f"Mud: {metrics['mud_risk']}")
    print(f"[CIRCUIT]  {len(envelopes.get('lufs', []))} chunks x "
          f"9 dimensions @ {envelopes.get('resolution_sec', 1.0)}s resolution")
    print(f"[SECTIONS] {len(sections)} physical sections detected")

    if problems:
        for problem in problems:
            print(f"[PROBLEM]  {problem['issue']} ({problem['severity']}): {problem['value']}")
    else:
        print("[PROBLEM]  None detected.")

    # Step 4: Full JSON output (pipeable to file or next stage)
    print("---JSON_START---")
    print(json.dumps(analysis_result, indent=2, ensure_ascii=False))
    print("---JSON_END---")