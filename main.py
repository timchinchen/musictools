from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile as scipy_wavfile
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.app import upload_store


app = FastAPI(title="MusicTools API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB
ALLOWED_EXTENSIONS = {".mp3"}
UPLOADS_DIR = Path(__file__).resolve().parent.parent / "uploads"
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@dataclass
class TabNote:
    note: str
    string: int
    fret: int
    beat: float


@dataclass(frozen=True)
class RiffPreset:
    id: str
    label: str
    string_costs: dict[int, float]
    fret_soft_low: int
    fret_soft_high: int
    max_comfort_fret_jump: int
    allowed_strings: frozenset[int] | None


@dataclass(frozen=True)
class Placement:
    string: int
    fret: int


RIFF_PRESETS: dict[str, RiffPreset] = {
    "low_string_rock": RiffPreset(
        id="low_string_rock",
        label="Low-string rock",
        string_costs={6: 0.0, 5: 0.35, 4: 0.55, 3: 1.8, 2: 2.5, 1: 3.0},
        fret_soft_low=3,
        fret_soft_high=10,
        max_comfort_fret_jump=4,
        allowed_strings=frozenset({4, 5, 6}),
    ),
    "mid_neck_rock": RiffPreset(
        id="mid_neck_rock",
        label="Mid-neck rock",
        string_costs={4: 0.0, 5: 0.5, 3: 0.6, 6: 1.2, 2: 2.0, 1: 2.5},
        fret_soft_low=3,
        fret_soft_high=9,
        max_comfort_fret_jump=4,
        allowed_strings=None,
    ),
    "lead_solo": RiffPreset(
        id="lead_solo",
        label="Lead / solo",
        string_costs={1: 0.0, 2: 0.3, 3: 0.8, 4: 1.2, 5: 2.0, 6: 2.8},
        fret_soft_low=5,
        fret_soft_high=17,
        max_comfort_fret_jump=5,
        allowed_strings=None,
    ),
}

OPEN_STRING_MIDI = [64, 59, 55, 50, 45, 40]
MAX_FRET = 20

AUDIO_PREPROCESS_MODES = frozenset({"none", "hpss_harmonic", "guitar_focus"})


def _normalize_peak(y: np.ndarray, target_peak: float = 0.92) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    m = float(np.max(np.abs(y)) + 1e-12)
    if m > 0.99:
        return (y / m * target_peak).astype(np.float32)
    return y


def _bandpass_guitar_range(y: np.ndarray, sr: int, low_hz: float = 82.0, high_hz: float = 5200.0) -> np.ndarray:
    y64 = np.asarray(y, dtype=np.float64)
    nyq = 0.5 * float(sr)
    lo = max(low_hz / nyq, 0.001)
    hi = min(high_hz / nyq, 0.999)
    if lo >= hi:
        return y.astype(np.float32)
    sos = signal.butter(4, [lo, hi], btype="band", output="sos")
    y_f = signal.sosfiltfilt(sos, y64)
    return y_f.astype(np.float32)


def _apply_audio_preprocess(y: np.ndarray, sr: int, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Lightweight guitar-focused preprocessing (not full stem separation)."""
    key = (mode or "none").strip().lower().replace("-", "_")
    if key not in AUDIO_PREPROCESS_MODES:
        key = "none"
    meta: dict[str, Any] = {
        "requested": mode,
        "applied": key,
        "steps": [],
        "note": (
            "HPSS separates harmonic vs percussive energy (helps reduce kick/snare bleed into pitch). "
            "guitar_focus adds a band-pass (~82 Hz–5.2 kHz). Vocals and other harmonic instruments can still remain."
        ),
    }
    if key == "none":
        return y.astype(np.float32, copy=False), meta

    work = np.asarray(y, dtype=np.float64)
    if key in ("hpss_harmonic", "guitar_focus"):
        y_harm, _y_perc = librosa.effects.hpss(work, margin=(2.0, 4.0))
        work = np.asarray(y_harm, dtype=np.float64)
        meta["steps"].append("librosa.effects.hpss (harmonic component)")

    if key == "guitar_focus":
        work = _bandpass_guitar_range(work, sr).astype(np.float64)
        meta["steps"].append("bandpass ~82 Hz – 5.2 kHz (guitar fundamentals + harmonics)")

    out = _normalize_peak(work.astype(np.float32))
    meta["rms_after_preprocess"] = round(float(np.sqrt(np.mean(np.square(out.astype(np.float64))) + 1e-20)), 6)
    return out, meta


def _float_audio_to_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    y = np.clip(y, -1.0, 1.0)
    pcm = np.round(y * 32767.0).astype(np.int16)
    buf = BytesIO()
    scipy_wavfile.write(buf, int(sr), pcm)
    return buf.getvalue()


def _validate_upload(filename: str, size: int) -> None:
    lower = filename.lower()
    if not any(lower.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Only MP3 files are supported.")

    if size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size is {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
        )


def _fallback_analysis(file_bytes: bytes) -> dict[str, Any]:
    digest = hashlib.sha1(file_bytes).hexdigest()
    seed = int(digest[:6], 16)

    tempo = 120 + (seed % 36)  # 120-155 BPM
    key_options = ["E minor", "A minor", "D minor", "G major", "E major"]
    key = key_options[seed % len(key_options)]

    progressions = [
        ["Em", "C", "G", "D"],
        ["Am", "F", "C", "G"],
        ["D5", "C5", "G5", "A5"],
        ["E5", "G5", "A5", "C5"],
    ]
    chords = progressions[seed % len(progressions)]

    riff = ["E4", "G4", "A4", "B4", "A4", "G4", "E4", "D4"]
    if seed % 2 == 0:
        riff = ["A3", "C4", "D4", "E4", "D4", "C4", "A3", "G3"]

    tab_map = {
        "E4": (2, 5),
        "G4": (2, 8),
        "A4": (1, 5),
        "B4": (1, 7),
        "D4": (2, 3),
        "A3": (3, 2),
        "C4": (3, 5),
        "E4_alt": (2, 5),
        "G3": (3, 0),
    }

    tab_notes: list[TabNote] = []
    beat = 1.0
    for note in riff:
        string, fret = tab_map.get(note, (3, 5))
        tab_notes.append(TabNote(note=note, string=string, fret=fret, beat=beat))
        beat += 0.5

    return {
        "tempo_bpm": tempo,
        "key": key,
        "chords": chords,
        "riff_notes": riff,
        "tab_suggestions": [tn.__dict__ for tn in tab_notes],
        "riff_preset": "mid_neck_rock",
        "riff_preset_label": "Mid-neck rock (fallback)",
        "confidence": {
            "chords": 0.78,
            "riff_notes": 0.64,
            "tab_mapping": 0.70,
        },
        "note": "Placeholder analysis. Replace with real DSP/ML transcription in next iteration.",
        "audio_preprocess": "none",
        "audio_preprocess_label": "Full mix",
        "debug": {
            "analysis_mode": "fallback",
            "source": "hash-seeded placeholder",
            "input_bytes": len(file_bytes),
            "pipeline_steps": [
                "Fallback: no real DSP (decode/analysis failed or unavailable).",
                "Output is deterministic from file hash — not from audio content.",
            ],
        },
    }


def _load_uploaded_file_path(file_id: str) -> Path:
    safe_id = file_id.strip()
    if not safe_id:
        raise HTTPException(status_code=400, detail="file_id is required.")

    file_path = UPLOADS_DIR / f"{safe_id}.mp3"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded file not found.")

    return file_path


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _pitch_class_name(pc: int) -> str:
    return NOTE_NAMES[pc % 12]


def _midi_to_note_name(midi: float) -> str:
    rounded = int(np.round(midi))
    note = _pitch_class_name(rounded % 12)
    octave = (rounded // 12) - 1
    return f"{note}{octave}"


def _estimate_key_with_debug(chroma_mean: np.ndarray) -> tuple[str, dict[str, Any]]:
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    major_scores: list[float] = []
    minor_scores: list[float] = []
    for shift in range(12):
        major_scores.append(float(np.dot(chroma_mean, np.roll(major_profile, shift))))
        minor_scores.append(float(np.dot(chroma_mean, np.roll(minor_profile, shift))))

    best_major = int(np.argmax(major_scores))
    best_minor = int(np.argmax(minor_scores))
    maj_top = sorted(enumerate(major_scores), key=lambda x: -x[1])[:3]
    min_top = sorted(enumerate(minor_scores), key=lambda x: -x[1])[:3]

    if major_scores[best_major] >= minor_scores[best_minor]:
        key = f"{_pitch_class_name(best_major)} major"
        chosen = "major"
        chosen_root_pc = best_major
        winning = major_scores[best_major]
        runner_up_mode = "minor"
        runner_up_score = float(minor_scores[best_minor])
        top_same = [{"root": _pitch_class_name(i), "score": round(s, 4)} for i, s in maj_top]
    else:
        key = f"{_pitch_class_name(best_minor)} minor"
        chosen = "minor"
        chosen_root_pc = best_minor
        winning = minor_scores[best_minor]
        runner_up_mode = "major"
        runner_up_score = float(major_scores[best_major])
        top_same = [{"root": _pitch_class_name(i), "score": round(s, 4)} for i, s in min_top]

    debug = {
        "key_krumhansl_mode": chosen,
        "key_root_pitch_class": chosen_root_pc,
        "key_winning_score": round(float(winning), 4),
        "key_runner_up_mode": runner_up_mode,
        "key_runner_up_best_score": round(runner_up_score, 4),
        "key_top3_same_mode": top_same,
        "key_top3_alternate_mode": [{"root": _pitch_class_name(i), "score": round(s, 4)} for i, s in (min_top if chosen == "major" else maj_top)],
    }
    return key, debug


def _estimate_key(chroma_mean: np.ndarray) -> str:
    key, _ = _estimate_key_with_debug(chroma_mean)
    return key


def _build_chord_templates() -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    templates: list[np.ndarray] = []
    for root in range(12):
        maj = np.zeros(12)
        min_chord = np.zeros(12)
        pow5 = np.zeros(12)
        maj[[root, (root + 4) % 12, (root + 7) % 12]] = 1.0
        min_chord[[root, (root + 3) % 12, (root + 7) % 12]] = 1.0
        pow5[[root, (root + 7) % 12]] = 1.0
        labels.extend([f"{_pitch_class_name(root)}", f"{_pitch_class_name(root)}m", f"{_pitch_class_name(root)}5"])
        templates.extend([maj, min_chord, pow5])
    return labels, np.array(templates)


CHORD_LABELS, CHORD_TEMPLATES = _build_chord_templates()


def _detect_chords_by_beat(chroma: np.ndarray, beat_frames: np.ndarray) -> list[str]:
    if beat_frames.size < 2:
        energy = np.mean(chroma, axis=1)
        sims = CHORD_TEMPLATES @ energy
        return [CHORD_LABELS[int(np.argmax(sims))]]

    chords: list[str] = []
    for i in range(len(beat_frames) - 1):
        start, end = int(beat_frames[i]), int(beat_frames[i + 1])
        if end <= start:
            continue
        segment = np.mean(chroma[:, start:end], axis=1)
        sims = CHORD_TEMPLATES @ segment
        chords.append(CHORD_LABELS[int(np.argmax(sims))])

    compressed: list[str] = []
    for chord in chords:
        if not compressed or compressed[-1] != chord:
            compressed.append(chord)
    return compressed[:16] if compressed else []


def _chords_per_beat_raw(chroma: np.ndarray, beat_frames: np.ndarray) -> list[str]:
    if beat_frames.size < 2:
        energy = np.mean(chroma, axis=1)
        sims = CHORD_TEMPLATES @ energy
        return [CHORD_LABELS[int(np.argmax(sims))]]

    out: list[str] = []
    for i in range(len(beat_frames) - 1):
        start, end = int(beat_frames[i]), int(beat_frames[i + 1])
        if end <= start:
            continue
        segment = np.mean(chroma[:, start:end], axis=1)
        sims = CHORD_TEMPLATES @ segment
        out.append(CHORD_LABELS[int(np.argmax(sims))])
    return out


def _pyin_track(y: np.ndarray, sr: int, hop_length: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    y_harmonic = librosa.effects.harmonic(y)
    f0, voiced_flag, _ = librosa.pyin(
        y_harmonic,
        sr=sr,
        fmin=librosa.note_to_hz("E2"),
        fmax=librosa.note_to_hz("E6"),
        hop_length=hop_length,
    )
    return f0, voiced_flag


def _extract_riff_notes_onset(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    max_notes: int = 48,
) -> tuple[list[float], list[float], dict[str, Any]]:
    f0, voiced_flag = _pyin_track(y, sr, hop_length)
    if f0 is None or voiced_flag is None:
        return [], [], {"error": "pyin_failed"}

    n_frames = int(f0.shape[0])
    voiced_ratio = float(np.mean(voiced_flag.astype(np.float64)))
    y_h = librosa.effects.harmonic(y)
    onset_frames = librosa.onset.onset_detect(
        y=y_h,
        sr=sr,
        hop_length=hop_length,
        units="frames",
        backtrack=True,
        delta=0.07,
        pre_max=3,
        post_max=3,
    )
    peaks = sorted({0, *[min(max(0, int(x)), n_frames - 1) for x in np.asarray(onset_frames, dtype=int)]})
    if peaks[-1] < n_frames - 1:
        peaks.append(n_frames - 1)

    meta: dict[str, Any] = {
        "pyin_frames": n_frames,
        "onset_raw_count": int(np.asarray(onset_frames).size),
        "onset_boundary_count": len(peaks),
        "voiced_frame_ratio": round(voiced_ratio, 4),
        "segments_seen": 0,
        "segments_skipped_short": 0,
        "segments_skipped_unvoiced": 0,
        "segments_used": 0,
    }

    midi_list: list[float] = []
    times: list[float] = []
    for i in range(len(peaks) - 1):
        meta["segments_seen"] += 1
        start, end = peaks[i], peaks[i + 1]
        if end - start < 2:
            meta["segments_skipped_short"] += 1
            continue
        seg_f0 = f0[start:end]
        seg_v = voiced_flag[start:end]
        valid = seg_v & np.isfinite(seg_f0) & (seg_f0 > 0)
        if not np.any(valid):
            meta["segments_skipped_unvoiced"] += 1
            continue
        m_hz = float(np.median(seg_f0[valid]))
        midi = float(np.round(librosa.hz_to_midi(m_hz)))
        center = (start + end) // 2
        t_center = float(librosa.frames_to_time(center, sr=sr, hop_length=hop_length))
        midi_list.append(midi)
        times.append(t_center)
        meta["segments_used"] += 1
        if len(midi_list) >= max_notes:
            break
    return midi_list, times, meta


def _extract_riff_notes_streaming(y: np.ndarray, sr: int, hop_length: int) -> tuple[list[float], list[float], dict[str, Any]]:
    f0, voiced_flag = _pyin_track(y, sr, hop_length)
    if f0 is None or voiced_flag is None:
        return [], [], {"error": "pyin_failed"}

    voiced_ratio = float(np.mean(voiced_flag.astype(np.float64)))
    meta = {
        "pyin_frames": int(f0.shape[0]),
        "voiced_frame_ratio": round(voiced_ratio, 4),
        "streaming_events": 0,
    }

    midi = librosa.hz_to_midi(np.where(voiced_flag, f0, np.nan))
    clean_m: list[float] = []
    clean_t: list[float] = []
    last: float | None = None
    for idx, m in enumerate(midi):
        if np.isnan(m):
            continue
        v = float(np.round(float(m)))
        if last is None or abs(v - last) >= 1:
            t = float(librosa.frames_to_time(idx, sr=sr, hop_length=hop_length))
            clean_m.append(v)
            clean_t.append(t)
            last = v
            meta["streaming_events"] += 1
        if len(clean_m) >= 48:
            break
    return clean_m, clean_t, meta


def _merge_same_grid(
    midis: list[float],
    beats: list[float],
    times_sec: list[float] | None = None,
) -> tuple[list[float], list[float], list[float]]:
    if not midis:
        return [], [], []
    out_m: list[float] = []
    out_b: list[float] = []
    out_t: list[float] = []
    for idx, (m, b) in enumerate(zip(midis, beats)):
        if out_m and abs(out_b[-1] - b) < 1e-4 and out_m[-1] == m:
            continue
        out_m.append(m)
        out_b.append(b)
        if times_sec is not None and idx < len(times_sec):
            out_t.append(times_sec[idx])
        elif times_sec is not None:
            out_t.append(times_sec[-1] if times_sec else 0.0)
    if times_sec is None:
        return out_m, out_b, []
    return out_m, out_b, out_t


def _quantize_to_sixteenths(
    times_sec: list[float],
    tempo_bpm: float,
    beat_frames: np.ndarray,
    sr: int,
    hop_length: int,
) -> tuple[list[float], dict[str, Any]]:
    if not times_sec:
        return [], {"anchor": "none"}
    bpm = max(40.0, float(tempo_bpm))
    sixteenth_sec = (60.0 / bpm) / 4.0
    if beat_frames.size > 0:
        t0 = float(librosa.frames_to_time(int(beat_frames[0]), sr=sr, hop_length=hop_length))
        anchor = "first_beat_frame"
    else:
        t0 = times_sec[0]
        anchor = "first_event_time"
    beats_out: list[float] = []
    for t in times_sec:
        steps = int(round((t - t0) / sixteenth_sec))
        beats_out.append(1.0 + steps * 0.25)
    meta = {
        "quantize_anchor": anchor,
        "grid_t0_sec": round(t0, 4),
        "sixteenth_note_sec": round(sixteenth_sec, 5),
        "bpm_used": round(bpm, 2),
    }
    return beats_out, meta


def _placements_for_midi(midi: float, preset: RiffPreset) -> list[Placement]:
    out: list[Placement] = []
    for string_idx, open_midi in enumerate(OPEN_STRING_MIDI, start=1):
        if preset.allowed_strings is not None and string_idx not in preset.allowed_strings:
            continue
        fret = int(round(midi - open_midi))
        if 0 <= fret <= MAX_FRET:
            out.append(Placement(string_idx, fret))
    if not out:
        for string_idx, open_midi in enumerate(OPEN_STRING_MIDI, start=1):
            fret = int(round(midi - open_midi))
            if 0 <= fret <= MAX_FRET:
                out.append(Placement(string_idx, fret))
    return out


def _placement_base_cost(p: Placement, preset: RiffPreset) -> float:
    cost = float(preset.string_costs.get(p.string, 1.5))
    if p.fret < preset.fret_soft_low or p.fret > preset.fret_soft_high:
        cost += 2.0
    return cost


def _transition_cost(prev: Placement, nxt: Placement, preset: RiffPreset) -> float:
    fret_jump = abs(prev.fret - nxt.fret)
    string_jump = abs(prev.string - nxt.string)
    cost = fret_jump * 0.85 + string_jump * 0.45
    if fret_jump > preset.max_comfort_fret_jump:
        cost += (fret_jump - preset.max_comfort_fret_jump) * 0.55
    return cost


def _dp_map_midi_to_tab(
    midi_notes: list[float],
    beat_values: list[float],
    preset: RiffPreset,
) -> tuple[list[TabNote], dict[str, Any]]:
    if not midi_notes or len(midi_notes) != len(beat_values):
        return [], {"error": "length_mismatch"}

    fallback_preset = RIFF_PRESETS["mid_neck_rock"]
    cands: list[list[Placement]] = []
    for m in midi_notes:
        c = _placements_for_midi(m, preset)
        if not c:
            c = _placements_for_midi(m, fallback_preset)
        cands.append(c)

    n = len(midi_notes)
    inf = 1e18
    dp: list[list[float]] = [[inf] * len(cands[i]) for i in range(n)]
    back: list[list[int]] = [[-1] * len(cands[i]) for i in range(n)]

    for j, pl in enumerate(cands[0]):
        dp[0][j] = _placement_base_cost(pl, preset)

    for i in range(1, n):
        for j, pl in enumerate(cands[i]):
            base = _placement_base_cost(pl, preset)
            best_val = inf
            best_k = -1
            prev_list = cands[i - 1]
            for k, prev_pl in enumerate(prev_list):
                val = dp[i - 1][k] + _transition_cost(prev_list[k], pl, preset) + base
                if val < best_val:
                    best_val = val
                    best_k = k
            dp[i][j] = best_val
            back[i][j] = best_k

    best_j = int(np.argmin(dp[-1]))
    dp_total_cost = float(dp[-1][best_j])
    path_j: list[int] = [best_j]
    for i in range(n - 1, 0, -1):
        best_j = back[i][best_j]
        path_j.append(best_j)
    path_j.reverse()

    cand_lens = [len(c) for c in cands]
    meta = {
        "dp_total_cost": round(dp_total_cost, 4),
        "dp_notes": n,
        "placement_candidates_min": int(min(cand_lens)) if cand_lens else 0,
        "placement_candidates_max": int(max(cand_lens)) if cand_lens else 0,
        "placement_candidates_mean": round(float(np.mean(cand_lens)), 3) if cand_lens else 0.0,
    }

    tab: list[TabNote] = []
    for i, j in enumerate(path_j):
        pl = cands[i][j]
        tab.append(
            TabNote(
                note=_midi_to_note_name(midi_notes[i]),
                string=pl.string,
                fret=pl.fret,
                beat=beat_values[i],
            )
        )
    return tab[:32], meta


def _analyze_mp3_file(file_path: Path, riff_preset_id: str, audio_preprocess: str = "none") -> dict[str, Any]:
    y_full, sr = librosa.load(str(file_path), sr=22050, mono=True)
    if y_full.size == 0:
        raise ValueError("Audio decode produced empty signal.")

    y_work, preprocess_meta = _apply_audio_preprocess(y_full, sr, audio_preprocess)

    hop_length = 512
    tempo_raw, beat_frames = librosa.beat.beat_track(y=y_full, sr=sr, hop_length=hop_length)
    tempo_bpm = float(np.round(float(np.squeeze(tempo_raw)), 1))
    beat_frames = np.asarray(beat_frames)

    chroma = librosa.feature.chroma_cqt(y=y_work, sr=sr, hop_length=hop_length)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_norm = chroma_mean / (np.sum(chroma_mean) + 1e-9)
    key, key_debug = _estimate_key_with_debug(chroma_norm)
    chords = _detect_chords_by_beat(chroma, beat_frames)
    chords_raw = _chords_per_beat_raw(chroma, beat_frames)

    preset = RIFF_PRESETS.get(riff_preset_id, RIFF_PRESETS["mid_neck_rock"])

    riff_midi, riff_times, extract_meta = _extract_riff_notes_onset(y_work, sr, hop_length)
    extraction = "onset_pyin"
    if len(riff_midi) < 4:
        riff_midi, riff_times, extract_meta = _extract_riff_notes_streaming(y_work, sr, hop_length)
        extraction = "streaming_pyin"

    beat_values, quant_meta = _quantize_to_sixteenths(riff_times, tempo_bpm, beat_frames, sr, hop_length)
    pre_merge_n = len(riff_midi)
    riff_midi, beat_values, riff_times_merged = _merge_same_grid(riff_midi, beat_values, riff_times)
    post_merge_n = len(riff_midi)

    mapped_tab, dp_meta = _dp_map_midi_to_tab(riff_midi, beat_values, preset)
    if mapped_tab:
        dp_meta["fallback_preset_used_any"] = any(len(_placements_for_midi(m, preset)) == 0 for m in riff_midi)
    tab_suggestions = [tn.__dict__ for tn in mapped_tab]
    riff_notes = [_midi_to_note_name(m) for m in riff_midi[:24]]

    rms_full = float(np.sqrt(np.mean(np.square(y_full.astype(np.float64))) + 1e-20))
    rms_work = float(np.sqrt(np.mean(np.square(y_work.astype(np.float64))) + 1e-20))
    rms_db = float(20.0 * np.log10(rms_work + 1e-12))

    note_events: list[dict[str, Any]] = []
    for i in range(min(len(mapped_tab), 24)):
        tn = mapped_tab[i]
        t_aligned = float(riff_times_merged[i]) if i < len(riff_times_merged) else None
        note_events.append(
            {
                "i": i,
                "time_sec": round(t_aligned, 4) if t_aligned is not None else None,
                "midi": int(riff_midi[i]) if i < len(riff_midi) else None,
                "note": tn.note,
                "beat_grid": round(float(beat_values[i]), 2) if i < len(beat_values) else None,
                "string": tn.string,
                "fret": tn.fret,
            }
        )

    pipeline_steps = [
        "1. Load mono audio, fixed SR",
        "2. Optional: HPSS / band-pass preprocess -> y_work (pitch/chroma path)",
        "3. beat_track on FULL mix -> tempo + beat frame grid",
        "4. chroma_cqt on y_work -> key + per-beat chord templates",
        "5. onset_detect + pyin on y_work -> MIDI pitch",
        "6. Snap event times to 16th-note grid from first beat",
        "7. Merge duplicate grid + same MIDI",
        "8. DP map MIDI -> (string,fret) with riff preset costs",
    ]

    chord_timeline_sample = [
        {"beat_slice_index": j, "chord": c}
        for j, c in enumerate(chords_raw[:24])
    ]

    return {
        "tempo_bpm": tempo_bpm,
        "key": key,
        "chords": chords,
        "riff_notes": riff_notes,
        "tab_suggestions": tab_suggestions,
        "riff_preset": preset.id,
        "riff_preset_label": preset.label,
        "confidence": {
            "chords": 0.72 if len(chords) > 1 else 0.45,
            "riff_notes": 0.62 if tab_suggestions else 0.30,
            "tab_mapping": 0.70 if tab_suggestions else 0.20,
        },
        "note": "Onset + pyin notes, 16th-note grid, DP tab mapping with riff preset.",
        "audio_preprocess": preprocess_meta["applied"],
        "audio_preprocess_label": {
            "none": "Full mix",
            "hpss_harmonic": "Harmonic only (HPSS)",
            "guitar_focus": "Guitar focus (HPSS + band-pass)",
        }.get(preprocess_meta["applied"], preprocess_meta["applied"]),
        "debug": {
            "analysis_mode": "real",
            "audio_preprocess": preprocess_meta,
            "pipeline_steps": pipeline_steps,
            "riff_preset": preset.id,
            "pitch_extraction": extraction,
            "pitch_extraction_detail": extract_meta,
            "sixteenth_quantize": True,
            "quantize": quant_meta,
            "merge_grid": {
                "events_before_merge": pre_merge_n,
                "events_after_merge": post_merge_n,
            },
            "tab_mapper": "dp_global",
            "dp": dp_meta,
            "audio": {
                "rms_linear_after_preprocess": round(rms_work, 6),
                "rms_linear_full_mix": round(rms_full, 6),
                "rms_dbfs_approx": round(rms_db, 2),
                "samples": int(y_full.size),
            },
            "key_detail": key_debug,
            "chord_timeline_sample": chord_timeline_sample,
            "chord_beat_slices_total": len(chords_raw),
            "note_events_sample": note_events,
            "sample_rate": sr,
            "hop_length": hop_length,
            "duration_seconds": float(np.round(librosa.get_duration(y=y_full, sr=sr), 2)),
            "beat_frame_count": int(beat_frames.size),
            "estimated_beat_count": max(int(beat_frames.size - 1), 0),
            "riff_candidate_count": int(len(riff_midi)),
            "tab_note_count": int(len(mapped_tab)),
            "chord_unique_count": int(len(chords)),
        },
    }


@app.get("/api/isolated-audio")
def isolated_audio(
    file_id: str,
    audio_preprocess: str = Query(
        default="guitar_focus",
        description="hpss_harmonic | guitar_focus (same as analysis)",
    ),
) -> Response:
    """WAV of the preprocessed signal (matches analysis path) for browser playback."""
    preprocess_key = audio_preprocess.strip().lower().replace("-", "_")
    if preprocess_key == "none":
        raise HTTPException(
            status_code=400,
            detail="audio_preprocess cannot be none; use hpss_harmonic or guitar_focus.",
        )
    if preprocess_key not in AUDIO_PREPROCESS_MODES:
        preprocess_key = "guitar_focus"

    file_path = _load_uploaded_file_path(file_id)
    y_full, sr = librosa.load(str(file_path), sr=22050, mono=True)
    if y_full.size == 0:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    y_work, _meta = _apply_audio_preprocess(y_full, sr, preprocess_key)
    wav_bytes = _float_audio_to_wav_bytes(y_work, sr)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="isolated.wav"'},
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/frontend/index.html")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    content = await file.read()
    original_name = file.filename or "unknown.mp3"
    _validate_upload(original_name, len(content))

    file_id = uuid.uuid4().hex
    file_path = UPLOADS_DIR / f"{file_id}.mp3"

    try:
        file_path.write_bytes(content)
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to store uploaded file.") from exc

    safe_name = os.path.basename(original_name)
    try:
        upload_store.record_upload(file_id, safe_name, len(content))
    except Exception:
        pass

    return {
        "file_id": file_id,
        "filename": safe_name,
        "size_bytes": len(content),
        "status": "uploaded",
    }


@app.get("/api/uploads")
async def list_upload_history(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
    """Recent uploads (newest first) for history UI."""
    try:
        items = upload_store.list_uploads(limit=limit)
    except Exception:
        items = []
    return {"uploads": items}


@app.post("/api/analyze")
async def analyze(
    file_id: str,
    riff_preset: str = Query(
        default="mid_neck_rock",
        description="low_string_rock | mid_neck_rock | lead_solo",
    ),
    audio_preprocess: str = Query(
        default="none",
        description="none | hpss_harmonic | guitar_focus",
    ),
) -> dict[str, Any]:
    file_path = _load_uploaded_file_path(file_id)
    preset_key = riff_preset.strip().lower().replace("-", "_")
    if preset_key not in RIFF_PRESETS:
        preset_key = "mid_neck_rock"
    preprocess_key = audio_preprocess.strip().lower().replace("-", "_")
    if preprocess_key not in AUDIO_PREPROCESS_MODES:
        preprocess_key = "none"
    try:
        result = _analyze_mp3_file(file_path, preset_key, preprocess_key)
    except Exception:
        # Keep API usable even when analysis fails on a specific file/backend codec.
        fallback_bytes = file_path.read_bytes()
        result = _fallback_analysis(fallback_bytes)
        result["note"] = (
            "Fallback analysis used. Install/verify audio codec support for more accurate results."
        )
        result.setdefault("debug", {})
        result["debug"]["error"] = "real_analysis_failed"
    result["file_id"] = file_id
    return result
