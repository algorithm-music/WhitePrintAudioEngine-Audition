# pyre-ignore-all-errors

"""Audio Analysis Service - Signal Processing & Feature Extraction.

Performs BS.1770-4 compliant audio analysis with Gemini semantic extraction.
"""

import gc
import json
import os
import struct
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import typing_extensions as typing
from google import genai
from google.genai import types as genai_types
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.signal import butter, resample_poly, sosfilt

LOG_FLOOR = 1e-10
A4_HZ = 440.0
TIME_SERIES_RESOLUTION_SEC = 0.1

BAND_EDGES = {
    "sub": (20.0, 60.0),
    "bass": (60.0, 200.0),
    "low_mid": (200.0, 500.0),
    "mid": (500.0, 2000.0),
    "high": (2000.0, 8000.0),
    "air": (8000.0, None),
}

MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
     2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
     2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
NOTE_NAMES = (
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
)


@dataclass(frozen=True)
class TrackSpectrum:
    """Pre-computed FFT data."""

    freqs: NDArray[np.float64]
    mono_power: NDArray[np.float64]
    side_power: NDArray[np.float64]

    @classmethod
    def compute(
        cls,
        mono: NDArray[np.float64],
        side: NDArray[np.float64],
        sr: int,
    ) -> "TrackSpectrum":
        # Compute FFT power spectra as float32 to halve peak memory
        # (rfft defaults to complex128; |.|**2 would be float64).
        mf = np.fft.rfft(mono)
        mp = (mf.real * mf.real + mf.imag * mf.imag).astype(
            np.float32, copy=False
        )
        del mf
        sf_ = np.fft.rfft(side)
        sp = (sf_.real * sf_.real + sf_.imag * sf_.imag).astype(
            np.float32, copy=False
        )
        del sf_
        f = np.fft.rfftfreq(len(mono), 1.0 / sr).astype(
            np.float32, copy=False
        )
        return cls(f, mp, sp)


class SectionSemanticContext(typing.TypedDict):
    """AI-driven section with timestamps and semantics."""
    section_name: str
    start_sec: float
    end_sec: float
    primary_instruments: List[str]
    musical_scene: str
    genre_foundation: str
    rhythmic_density: str


class MacroFormResponse(typing.TypedDict):
    """Top-level Gemini response for full song structure."""
    sections: List[SectionSemanticContext]


def _build_k_weight_sos(sr: int) -> NDArray:
    """BS.1770-4 K-weighting filter."""
    f0 = 1681.974450955533
    gain = 4.0
    q = 0.7071752369554196
    w0 = 2.0 * np.pi * f0 / sr
    a_v = 10.0 ** (gain / 40.0)
    alpha = np.sin(w0) / (2.0 * q)
    cw = np.cos(w0)
    sa = np.sqrt(a_v)

    b0 = a_v * ((a_v + 1) + (a_v - 1) * cw + 2 * sa * alpha)
    b1 = -2 * a_v * ((a_v - 1) + (a_v + 1) * cw)
    b2 = a_v * ((a_v + 1) + (a_v - 1) * cw - 2 * sa * alpha)
    a0 = (a_v + 1) - (a_v - 1) * cw + 2 * sa * alpha
    a1 = 2 * ((a_v - 1) - (a_v + 1) * cw)
    a2 = (a_v + 1) - (a_v - 1) * cw - 2 * sa * alpha

    hs = np.array(
        [b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]
    ).reshape(1, 6)
    hp = butter(2, 38.0, btype="highpass", fs=sr, output="sos")
    return np.vstack([hs, hp])


def validate_audio_file(fp: str) -> Dict[str, Any]:
    """Validate audio file headers."""
    if os.path.getsize(fp) < 44:
        raise ValueError("File too small.")
    try:
        info = sf.info(fp)
    except Exception:
        raise ValueError("Unsupported format.")
    if info.samplerate < 8000 or info.samplerate > 384000:
        raise ValueError(f"Bad sample rate: {info.samplerate}Hz")
    return {
        "duration_sec": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
    }


def analyze_audio_file(fp: str) -> Dict[str, Any]:
    """Main analysis entry point."""
    data, sr = sf.read(fp, dtype="float32")
    if data.ndim == 1:
        left = np.ascontiguousarray(data, dtype=np.float32)
        right = left
    else:
        left = np.ascontiguousarray(data[:, 0], dtype=np.float32)
        right = np.ascontiguousarray(data[:, 1], dtype=np.float32)
    duration_sec = round(float(len(left) / sr), 2)
    del data
    gc.collect()

    mono = ((left + right) * 0.5).astype(np.float32, copy=False)
    side = ((left - right) * 0.5).astype(np.float32, copy=False)

    spec = TrackSpectrum.compute(mono, side, sr)
    k_sos = _build_k_weight_sos(sr)
    lk = sosfilt(k_sos, left).astype(np.float32, copy=False)
    rk = sosfilt(k_sos, right).astype(np.float32, copy=False)
    mk = ((lk + rk) * 0.5).astype(np.float32, copy=False)
    gc.collect()

    envs = _compute_envelopes(mono, mk, side, left, right, sr)
    del mk
    gc.collect()

    metrics = _compute_metrics(
        mono, left, right, mono, side, lk, rk, sr, spec, envs
    )
    # lk, rk, spec no longer needed past this point.
    del lk, rk, spec
    gc.collect()

    bpm = _estimate_bpm(mono, sr)
    key = _estimate_key(mono, sr)
    bit_depth = _detect_bit_depth(fp)

    sections = _detect_sections(mono, sr, envs)

    return {
        "track_identity": {
            "duration_sec": duration_sec,
            "sample_rate": sr,
            "bpm": bpm,
            "key": key,
            "bit_depth": bit_depth,
        },
        "whole_track_metrics": metrics,
        "time_series_circuit_envelopes": envs,
        "physical_sections": sections,
        "detected_problems": _detect_problems(metrics),
    }


def _compute_envelopes(
    mono: NDArray, mk: NDArray, side: NDArray,
    left: NDArray, right: NDArray, sr: int,
) -> Dict[str, Any]:
    """9-dimensional time-series envelopes."""
    cs = int(sr * TIME_SERIES_RESOLUTION_SEC)
    nc = len(mono) // cs
    if nc == 0:
        return {}

    vl = nc * cs
    m_ch = mono[:vl].reshape(nc, cs)
    mk_ch = mk[:vl].reshape(nc, cs)
    s_ch = side[:vl].reshape(nc, cs)
    l_ch = left[:vl].reshape(nc, cs)
    r_ch = right[:vl].reshape(nc, cs)

    msq = np.mean(mk_ch ** 2, axis=1) + LOG_FLOOR
    lufs = -0.691 + 10.0 * np.log10(msq)

    rms = np.sqrt(np.mean(m_ch ** 2, axis=1) + LOG_FLOOR)
    pk = np.max(np.abs(m_ch), axis=1)
    crest = (
        20.0 * np.log10(np.maximum(pk, LOG_FLOOR))
        - 20.0 * np.log10(rms)
    )

    srms = np.sqrt(np.mean(s_ch ** 2, axis=1) + LOG_FLOOR)
    width = np.clip(srms / rms, 0.0, 1.0)

    pwr = np.abs(np.fft.rfft(m_ch, axis=1)) ** 2
    freqs = np.fft.rfftfreq(cs, 1.0 / sr)
    te = np.sum(pwr[:, freqs >= 20], axis=1) + LOG_FLOOR

    sub_r = np.sum(
        pwr[:, (freqs >= 20) & (freqs < 60)], axis=1
    ) / te
    bass_r = np.sum(
        pwr[:, (freqs >= 60) & (freqs < 200)], axis=1
    ) / te
    voc_r = np.sum(
        pwr[:, (freqs >= 1000) & (freqs < 5000)], axis=1
    ) / te
    wf = np.sum(pwr * freqs, axis=1)
    bright = wf / te / (cs // 2)

    lp = butter(4, 120.0, btype="lowpass", fs=sr, output="sos")
    ll = sosfilt(lp, l_ch, axis=1).astype(np.float32, copy=False)
    rl = sosfilt(lp, r_ch, axis=1).astype(np.float32, copy=False)
    llm = np.mean(ll, axis=1, keepdims=True)
    rlm = np.mean(rl, axis=1, keepdims=True)
    # Center in place, free the mean offsets, and compute sums via einsum
    # to avoid materializing (ll-llm)**2 and (rl-rlm)**2 full arrays.
    ll -= llm
    rl -= rlm
    del llm, rlm
    num = np.einsum("ij,ij->i", ll, rl)
    dl = np.einsum("ij,ij->i", ll, ll)
    dr = np.einsum("ij,ij->i", rl, rl)
    del ll, rl
    den = np.sqrt(dl * dr) + LOG_FLOOR
    lmc = np.clip(num / den, -1.0, 1.0)
    del num, dl, dr, den

    tr = np.zeros(nc)
    for i in range(nc):
        tr[i] = float(np.percentile(np.abs(np.diff(m_ch[i])), 95))

    def _r(a, d):
        return [round(float(v), d) for v in a]

    return {
        "resolution_sec": TIME_SERIES_RESOLUTION_SEC,
        "lufs": _r(lufs, 1),
        "crest_db": _r(crest, 1),
        "width": _r(width, 3),
        "sub_ratio": _r(sub_r, 3),
        "bass_ratio": _r(bass_r, 3),
        "vocal_presence": _r(voc_r, 3),
        "spectral_brightness": _r(bright, 4),
        "low_mono_correlation": _r(lmc, 3),
        "transient_sharpness": _r(tr, 6),
    }


def _compute_metrics(
    mono: NDArray, left: NDArray, right: NDArray,
    mid: NDArray, side: NDArray,
    lk: NDArray, rk: NDArray,
    sr: int, spec: TrackSpectrum,
    envs: Dict[str, Any],
) -> Dict[str, Any]:
    """BS.1770-4 whole-track metrics."""
    # einsum computes sum-of-squares without materializing lk**2 / rk**2.
    mkp = (
        float(np.einsum("i,i->", lk, lk)) / len(lk)
        + float(np.einsum("i,i->", rk, rk)) / len(rk)
    )
    i_lufs = -0.691 + 10.0 * np.log10(max(mkp, LOG_FLOOR))

    tp_v = _true_peak_chunked(left, right)
    tp_db = 20.0 * np.log10(max(tp_v, LOG_FLOOR))

    st_vals = []
    ws = int(3.0 * sr)
    hs = int(1.0 * sr)
    for si in range(0, len(lk) - ws, hs):
        ei = si + ws
        pl = np.mean(lk[si:ei] ** 2)
        pr = np.mean(rk[si:ei] ** 2)
        st = -0.691 + 10.0 * np.log10(max(pl + pr, LOG_FLOOR))
        st_vals.append(st)

    sta = np.array(st_vals)
    gst = sta[sta > -70.0]

    if len(gst) > 0:
        mp = max(np.mean(10 ** ((gst + 0.691) / 10.0)), LOG_FLOOR)
        rt = -0.691 + 10.0 * np.log10(mp) - 20.0
        fg = gst[gst > rt]
        lra = (
            float(np.percentile(fg, 95) - np.percentile(fg, 10))
            if len(fg) > 2 else 0.0
        )
        stm = float(np.max(gst))
    else:
        lra = 0.0
        stm = i_lufs

    psr = tp_db - stm

    te = np.sum(spec.mono_power[spec.freqs >= 20]) + LOG_FLOOR
    ratios = {}
    for bn, (lo, hi) in BAND_EDGES.items():
        hi_v = hi if hi is not None else np.inf
        mask = (spec.freqs >= lo) & (spec.freqs < hi_v)
        ratios[f"{bn}_ratio"] = round(
            float(np.sum(spec.mono_power[mask]) / te), 4
        )

    hm = (spec.freqs >= 2000) & (spec.freqs < 6000)
    hr = np.clip(
        (np.sum(spec.mono_power[hm]) / te) * 3.0, 0.0, 1.0
    )
    mr = np.clip(
        (ratios.get("low_mid_ratio", 0.0) - 0.15) / 0.15,
        0.0, 1.0,
    )

    m_rms = np.sqrt(np.mean(mono ** 2))
    m_pk = np.max(np.abs(mono))
    cdb = (
        20.0 * np.log10(max(m_pk, LOG_FLOOR))
        - 20.0 * np.log10(max(m_rms, LOG_FLOOR))
    )

    s_rms = np.sqrt(np.mean(side ** 2))
    mid_rms = np.sqrt(np.mean(mid ** 2))
    sw = np.clip(s_rms / (mid_rms + LOG_FLOOR), 0.0, 1.0)

    sc = np.corrcoef(left, right)[0, 1]
    lmc = float(np.mean(envs.get("low_mono_correlation", [1.0])))

    return {
        "integrated_lufs": round(float(i_lufs), 1),
        "true_peak_dbtp": round(float(tp_db), 2),
        "lra_lu": round(float(lra), 1),
        "psr_db": round(float(psr), 1),
        "crest_db": round(float(cdb), 1),
        "stereo_width": round(float(sw), 3),
        "stereo_correlation": round(float(sc), 3),
        "low_mono_correlation_below_120hz": round(lmc, 3),
        "harshness_risk": round(float(hr), 3),
        "mud_risk": round(float(mr), 3),
        **ratios,
    }


def _true_peak_chunked(left: NDArray, right: NDArray) -> float:
    """Chunked 4x oversampled true peak."""
    cs = 44100 * 5
    gm = max(np.max(np.abs(left)), np.max(np.abs(right)))
    if gm < 1e-4:
        return float(gm)

    tpm = 0.0
    for si in range(0, len(left), cs):
        ei = si + cs
        lc = left[si:ei]
        rc = right[si:ei]
        cm = max(np.max(np.abs(lc)), np.max(np.abs(rc)))
        if cm > gm * 0.5:
            lo = resample_poly(lc, 4, 1)
            ro = resample_poly(rc, 4, 1)
            tpm = max(tpm, np.max(np.abs(lo)), np.max(np.abs(ro)))
    return float(max(tpm, gm))


def _extract_macro_form(
    mono: NDArray, sr: int, duration: float,
) -> Optional[List[Dict[str, Any]]]:
    """Listen to full track once with Gemini on Vertex AI, get sections + semantics.

    Uses GCSFuse mount to stream audio to GCS, then passes gs:// URI to Vertex
    (avoids loading audio into Python process memory).
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "asia-northeast1")
    gcs_mount = os.environ.get("GCSFUSE_MOUNT", "/mnt/gcs/aimastering-tmp-audio")
    gcs_bucket = os.environ.get("GCSFUSE_BUCKET", "aidriven-mastering-fyqu-source-bucket")
    if not project:
        print("[WARN] GOOGLE_CLOUD_PROJECT not set; skipping Gemini extraction.")
        return None
    if not os.path.isdir(gcs_mount):
        print(f"[WARN] GCSFuse mount missing at {gcs_mount}; skipping Gemini extraction.")
        return None

    client = genai.Client(vertexai=True, project=project, location=location)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    # Downsample to 16kHz mono for fast upload
    target_sr = 16000
    down = resample_poly(mono, target_sr, sr)

    object_name = f"vertex-audio-{uuid.uuid4().hex}.wav"
    fuse_path = os.path.join(gcs_mount, object_name)
    gs_uri = f"gs://{gcs_bucket}/{object_name}"

    try:
        # Writing through the GCSFuse mount streams to GCS without buffering in memory.
        sf.write(fuse_path, down, target_sr)

        prompt = (
            f"Listen to this track ({duration:.1f}s). "
            "Divide it into main musical sections "
            "(Intro, Verse, Build-up, Drop, Breakdown, Outro, etc). "
            "RULES: "
            "1. First section starts at 0.0. "
            f"2. Last section ends at {duration:.1f}. "
            "3. No section shorter than 15 seconds. "
            "4. Be objective and precise."
        )
        resp = client.models.generate_content(
            model=model_name,
            contents=[
                prompt,
                genai_types.Part.from_uri(
                    file_uri=gs_uri, mime_type="audio/wav"
                ),
            ],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MacroFormResponse,
                temperature=0.0,
            ),
        )
        result = json.loads(resp.text)
        return result.get("sections", [])
    except Exception as e:
        print(f"[WARN] Gemini macro-form failed: {e}")
        return None
    finally:
        try:
            if os.path.exists(fuse_path):
                os.remove(fuse_path)
        except Exception:
            pass


def _detect_sections(
    mono: NDArray, sr: int, envs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """AI-driven segmentation with DSP fallback."""
    lufs_e = np.array(envs.get("lufs", []))
    width_e = np.array(envs.get("width", []))
    duration = float(len(mono) / sr)
    res = TIME_SERIES_RESOLUTION_SEC

    # 1. Try Gemini (1 API call for entire track)
    ai_secs = _extract_macro_form(mono, sr, duration)

    if ai_secs and len(ai_secs) > 0:
        sections = []
        for idx, sec in enumerate(ai_secs):
            ss = max(0.0, float(sec.get("start_sec", 0)))
            es = min(duration, float(sec.get("end_sec", duration)))
            if es - ss < 1.0:
                continue

            si = int(ss / res)
            ei = min(int(es / res), len(lufs_e))
            al = (
                round(float(np.mean(lufs_e[si:ei])), 1)
                if si < ei and ei <= len(lufs_e) else -14.0
            )
            aw = (
                round(float(np.mean(width_e[si:ei])), 3)
                if si < ei and ei <= len(width_e) else 0.0
            )

            name = sec.get("section_name", "Part")
            ctx = {
                "primary_instruments": sec.get("primary_instruments", []),
                "musical_scene": sec.get("musical_scene", ""),
                "genre_foundation": sec.get("genre_foundation", ""),
                "rhythmic_density": sec.get("rhythmic_density", ""),
            }

            sections.append({
                "section_id": f"SEC_{idx}_{name.replace(' ', '')}",
                "start_sec": round(ss, 2),
                "end_sec": round(es, 2),
                "avg_lufs": al,
                "avg_width": aw,
                "semantic_context": ctx,
            })
        if sections:
            return sections

    # 2. DSP fallback (no Gemini)
    return _dsp_fallback(lufs_e, width_e, duration)


def _dsp_fallback(
    lufs_e: NDArray, width_e: NDArray, duration: float,
) -> List[Dict[str, Any]]:
    """DSP novelty-based segmentation (15s min)."""
    if len(lufs_e) < 100:
        return [{
            "section_id": "SEC_0_Full",
            "start_sec": 0.0,
            "end_sec": duration,
            "avg_lufs": round(float(np.mean(lufs_e)), 1)
            if len(lufs_e) > 0 else -14.0,
            "avg_width": round(float(np.mean(width_e)), 3)
            if len(width_e) > 0 else 0.0,
            "semantic_context": None,
        }]

    sw = int(3.0 / TIME_SERIES_RESOLUTION_SEC)
    w = np.ones(sw) / sw
    sl = np.convolve(lufs_e, w, mode="same")
    swi = np.convolve(width_e, w, mode="same")
    ld = np.abs(np.diff(sl))
    wd = np.abs(np.diff(swi))
    ln = ld / (np.max(ld) + 1e-6)
    wn = wd / (np.max(wd) + 1e-6)
    nov = ln + (wn * 0.3)

    th = np.mean(nov) + 1.2 * np.std(nov)
    bounds = [0]
    mc = int(15.0 / TIME_SERIES_RESOLUTION_SEC)

    for i in range(1, len(nov) - 1):
        if (
            nov[i] > th
            and nov[i] > nov[i - 1]
            and nov[i] > nov[i + 1]
        ):
            if (i - bounds[-1]) > mc:
                bounds.append(i)

    total = len(lufs_e)
    if total - bounds[-1] < mc and len(bounds) > 1:
        bounds[-1] = total
    else:
        bounds.append(total)

    res = TIME_SERIES_RESOLUTION_SEC
    secs = []
    for idx in range(len(bounds) - 1):
        si = bounds[idx]
        ei = bounds[idx + 1]
        secs.append({
            "section_id": f"SEC_{idx}_Part",
            "start_sec": round(float(si * res), 2),
            "end_sec": round(float(ei * res), 2),
            "avg_lufs": round(float(np.mean(lufs_e[si:ei])), 1),
            "avg_width": round(float(np.mean(width_e[si:ei])), 3),
            "semantic_context": None,
        })
    return secs


def _estimate_bpm(mono: NDArray, sr: int) -> Optional[float]:
    """Full-track BPM estimation.

    Uses a chunked einsum to compute per-frame energy without materializing
    the (n_frames, 1024) squared-frame array, which would be hundreds of MB
    for a multi-minute track.
    """
    frames = sliding_window_view(mono, window_shape=1024)[::512]
    n_frames = len(frames)
    if n_frames == 0:
        return None
    energy = np.empty(n_frames, dtype=np.float32)
    chunk = 8192
    for i in range(0, n_frames, chunk):
        end = min(i + chunk, n_frames)
        blk = frames[i:end]
        energy[i:end] = np.einsum("ij,ij->i", blk, blk).astype(
            np.float32, copy=False
        )
    onsets = np.maximum(np.diff(energy), 0)
    del energy
    corr = np.correlate(onsets, onsets, mode="full")
    corr = corr[len(onsets) - 1:]
    del onsets
    fps = sr / 512.0
    lo = int(fps * 60 / 200)
    hi = int(fps * 60 / 60)
    if hi > len(corr) or len(corr[lo:hi]) == 0:
        return None
    bl = np.argmax(corr[lo:hi]) + lo
    return round(float(fps * 60.0 / bl), 1)


def _estimate_key(mono: NDArray, sr: int) -> Optional[str]:
    """Full-track key detection."""
    # Compute magnitude spectrum as float32 to match TrackSpectrum path.
    mf = np.fft.rfft(mono)
    fft_r = np.sqrt(
        (mf.real * mf.real + mf.imag * mf.imag).astype(
            np.float32, copy=False
        )
    )
    del mf
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr).astype(
        np.float32, copy=False
    )
    chroma = np.zeros(12)
    for note in np.arange(24, 96):
        tf = A4_HZ * 2 ** ((note - 69) / 12)
        idx = int(np.clip(
            np.searchsorted(freqs, tf), 0, len(freqs) - 1
        ))
        chroma[note % 12] += float(fft_r[idx]) ** 2
    del fft_r, freqs
    gc.collect()

    bc, bk, bm = -1.0, "C", "major"
    for shift in range(12):
        sc = np.roll(chroma, -shift)
        cm = np.corrcoef(sc, MAJOR_PROFILE)[0, 1]
        cn = np.corrcoef(sc, MINOR_PROFILE)[0, 1]
        if cm > bc:
            bc, bk, bm = cm, NOTE_NAMES[shift], "major"
        if cn > bc:
            bc, bk, bm = cn, NOTE_NAMES[shift], "minor"
    return f"{bk} {bm}"


def _detect_problems(
    wm: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Detect engineering issues."""
    p = []
    if wm.get("true_peak_dbtp", -100.0) > -0.3:
        p.append({
            "issue": "true_peak_danger",
            "severity": "high",
            "value": wm["true_peak_dbtp"],
        })
    if wm.get("mud_risk", 0.0) > 0.4:
        p.append({
            "issue": "mud_risk",
            "severity": "medium",
            "value": wm["mud_risk"],
        })
    if wm.get("low_mono_correlation_below_120hz", 1.0) < 0.3:
        p.append({
            "issue": "phase_cancellation_lows",
            "severity": "medium",
            "value": wm["low_mono_correlation_below_120hz"],
        })
    if wm.get("harshness_risk", 0.0) > 0.5:
        p.append({
            "issue": "harshness_risk",
            "severity": "medium",
            "value": wm["harshness_risk"],
        })
    if wm.get("crest_db", 20.0) < 3.0:
        p.append({
            "issue": "extremely_low_crest_factor",
            "severity": "low",
            "value": wm["crest_db"],
        })
    return p


def _detect_bit_depth(fp: str) -> int:
    """Detect WAV bit depth from RIFF header."""
    try:
        with open(fp, "rb") as f:
            h = f.read(40)
        if len(h) >= 36 and h[:4] == b"RIFF":
            return struct.unpack_from("<H", h, 34)[0]
    except Exception:
        pass
    return 24


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python audio_analysis.py <file>")
        sys.exit(1)

    path = sys.argv[1]
    if "--validate-only" in sys.argv:
        print(validate_audio_file(path))
        sys.exit(0)

    if not (
        os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    ):
        print("[NOTICE] No Gemini credentials. DSP-only mode.")

    result = analyze_audio_file(path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
