"""
Violence Detection Dashboard - Flask Backend (v2)
==================================================
Fixes:
1. Process-level CPU/RAM tracking (isolated from background apps)
2. 5-second video segmentation for correct predictions on long clips
"""

import os
import sys
import json
import time
import threading
import queue
import psutil
import numpy as np
import cv2

# ── Suppress TF noise ──────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from predict_video import (
    load_video,
    preprocess_for_model,
    load_model,
    predict,
    compute_optical_flow,
    uniform_sampling,
    normalize,
)

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="static")
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Models", "keras_model.h5"
)

_model = None
_model_lock = threading.Lock()
_sse_queues: dict[str, queue.Queue] = {}

MOTION_GATE_THRESHOLD = 15.0
SEGMENT_DURATION_SEC = 5.0  # Split video into 5s chunks (matches RWF-2000)


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            _model = load_model(MODEL_PATH)
    return _model


# ── Motion-energy gate ─────────────────────────────────────────────────────────
def motion_energy_gate(frames_bgr, sample_count=8, gate_resolution=(56, 56)):
    n = len(frames_bgr)
    if n < 2:
        return 0.0
    indices = np.linspace(0, n - 1, min(sample_count + 1, n), dtype=int)
    diffs = []
    for i in range(len(indices) - 1):
        f1 = cv2.resize(frames_bgr[indices[i]], gate_resolution)
        f2 = cv2.resize(frames_bgr[indices[i + 1]], gate_resolution)
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diffs.append(np.mean(np.abs(g1 - g2)))
    return float(np.mean(diffs))


# ── Process-level CPU sampler ──────────────────────────────────────────────────
def cpu_sampler(job_id: str, stop_event: threading.Event, interval: float = 0.4):
    """
    Tracks CPU/RAM of THIS process only (not system-wide).
    Also tracks cumulative CPU-time (user+system seconds) for deterministic measurement.
    """
    q = _sse_queues[job_id]
    proc = psutil.Process(os.getpid())
    # Prime the cpu_percent call (first call always returns 0)
    proc.cpu_percent(interval=None)
    time.sleep(0.1)

    cpu_count = psutil.cpu_count() or 1

    while not stop_event.is_set():
        try:
            # Process-level CPU % (can exceed 100% on multi-core, normalize)
            raw_cpu = proc.cpu_percent(interval=None)
            cpu_pct = min(raw_cpu / cpu_count, 100.0)

            mem_info = proc.memory_info()
            mem_mb = mem_info.rss / (1024 * 1024)

            # CPU-time: deterministic, reproducible
            cpu_times = proc.cpu_times()
            cpu_time_s = cpu_times.user + cpu_times.system

            q.put({
                "type": "cpu",
                "cpu": round(cpu_pct, 1),
                "mem_mb": round(mem_mb, 1),
                "cpu_time_s": round(cpu_time_s, 2),
                "ts": time.time(),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        time.sleep(interval)


# ── Segment-level processing ───────────────────────────────────────────────────
def process_segment(seg_frames_bgr, model, use_gate, seg_index, seg_start_s, seg_end_s):
    """
    Process a single ~5-second segment. Returns a dict with results.
    """
    result = {
        "seg_index": seg_index,
        "start_s": round(seg_start_s, 2),
        "end_s": round(seg_end_s, 2),
        "frames": len(seg_frames_bgr),
        "gate_triggered": True,
        "energy": None,
        "gate_ms": 0,
        "optflow_ms": 0,
        "model_ms": 0,
        "total_ms": 0,
        "label": "NonFight (Calm)",
        "confidence": 1.0,
        "fight_prob": 0.0,
        "nonfight_prob": 1.0,
    }

    t_seg_start = time.perf_counter()

    # Gate
    if use_gate:
        t_g = time.perf_counter()
        energy = motion_energy_gate(seg_frames_bgr)
        result["gate_ms"] = round((time.perf_counter() - t_g) * 1000, 1)
        result["energy"] = round(energy, 3)

        if energy < MOTION_GATE_THRESHOLD:
            result["gate_triggered"] = False
            result["label"] = "NonFight (Calm - Skipped)"
            result["total_ms"] = round((time.perf_counter() - t_seg_start) * 1000, 1)
            return result

    # Resize to 224x224 RGB
    resized = []
    for f in seg_frames_bgr:
        resized.append(cv2.cvtColor(cv2.resize(f, (224, 224)), cv2.COLOR_BGR2RGB))
    resized = np.array(resized, dtype=np.uint8)

    # Optical flow
    t_f = time.perf_counter()
    flows = compute_optical_flow(resized)
    result["optflow_ms"] = round((time.perf_counter() - t_f) * 1000, 1)

    # Pack + preprocess
    data = np.zeros((len(resized), 224, 224, 5), dtype=np.float32)
    data[..., :3] = resized.astype(np.float32)
    data[..., 3:] = flows
    processed = preprocess_for_model(data)

    # Model inference
    t_m = time.perf_counter()
    label, confidence, probs = predict(model, processed)
    result["model_ms"] = round((time.perf_counter() - t_m) * 1000, 1)

    result["label"] = label
    result["confidence"] = round(float(confidence), 4)
    result["fight_prob"] = round(float(probs[0]), 4)
    result["nonfight_prob"] = round(float(probs[1]), 4)
    result["total_ms"] = round((time.perf_counter() - t_seg_start) * 1000, 1)
    return result


# ── Core processing logic (v2 — segmented) ────────────────────────────────────
def run_inference(job_id: str, video_path: str, use_gate: bool):
    q = _sse_queues[job_id]
    stop_cpu = threading.Event()

    def push(data: dict):
        q.put(data)

    try:
        # Start process-level CPU sampler
        cpu_thread = threading.Thread(
            target=cpu_sampler, args=(job_id, stop_cpu), daemon=True
        )
        cpu_thread.start()

        t_total_start = time.perf_counter()

        # ── Read all frames ────────────────────────────────────────────────────
        push({"type": "progress", "step": "reading", "msg": "Reading video frames…", "pct": 5})
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        frames_bgr = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_bgr.append(frame)
        cap.release()

        total_frames = len(frames_bgr)
        duration_video = total_frames / fps
        push({
            "type": "info",
            "frames": total_frames,
            "fps": round(fps, 1),
            "duration": round(duration_video, 2),
        })

        # ── Split into 5-second segments ───────────────────────────────────────
        frames_per_seg = int(SEGMENT_DURATION_SEC * fps)
        if frames_per_seg < 16:
            frames_per_seg = max(16, total_frames)  # safety for very short clips

        segments = []
        for start_idx in range(0, total_frames, frames_per_seg):
            end_idx = min(start_idx + frames_per_seg, total_frames)
            if end_idx - start_idx < 8:
                # Too short, merge into last segment
                if segments:
                    prev = segments[-1]
                    segments[-1] = (prev[0], end_idx)
                break
            segments.append((start_idx, end_idx))

        if not segments:
            segments = [(0, total_frames)]

        num_segments = len(segments)
        push({
            "type": "progress",
            "step": "segmenting",
            "msg": f"Split into {num_segments} segment(s) of ~{SEGMENT_DURATION_SEC:.0f}s each",
            "pct": 10,
        })

        # ── Process each segment ───────────────────────────────────────────────
        model = get_model()
        segment_results = []
        total_gate_ms = 0
        total_optflow_ms = 0
        total_model_ms = 0
        segments_skipped = 0
        any_fight = False

        for i, (s_start, s_end) in enumerate(segments):
            seg_frames = frames_bgr[s_start:s_end]
            seg_start_s = s_start / fps
            seg_end_s = s_end / fps

            pct_base = 10 + int(80 * i / num_segments)
            push({
                "type": "progress",
                "step": f"segment_{i}",
                "msg": f"Segment {i+1}/{num_segments} [{seg_start_s:.1f}s – {seg_end_s:.1f}s]…",
                "pct": pct_base,
            })

            seg_result = process_segment(seg_frames, model, use_gate, i, seg_start_s, seg_end_s)
            segment_results.append(seg_result)

            total_gate_ms += seg_result["gate_ms"]
            total_optflow_ms += seg_result["optflow_ms"]
            total_model_ms += seg_result["model_ms"]

            if not seg_result["gate_triggered"]:
                segments_skipped += 1

            is_fight = ("fight" in seg_result["label"].lower()
                        and "non" not in seg_result["label"].lower()
                        and "calm" not in seg_result["label"].lower())
            if is_fight:
                any_fight = True

            # Push per-segment result for timeline
            push({"type": "segment_result", **seg_result, "is_fight": is_fight})

        t_total = (time.perf_counter() - t_total_start) * 1000

        # ── Overall verdict ────────────────────────────────────────────────────
        if any_fight:
            fight_segments = [s for s in segment_results
                              if "fight" in s["label"].lower()
                              and "non" not in s["label"].lower()
                              and "calm" not in s["label"].lower()]
            best = max(fight_segments, key=lambda s: s["fight_prob"])
            overall_label = best["label"]
            overall_confidence = best["confidence"]
            overall_fight_prob = best["fight_prob"]
            overall_nonfight_prob = best["nonfight_prob"]
        else:
            overall_label = "NonFight (Non-Violence)"
            # average confidence of segments that ran the model
            model_segments = [s for s in segment_results if s["gate_triggered"]]
            if model_segments:
                overall_confidence = np.mean([s["confidence"] for s in model_segments])
                overall_fight_prob = np.mean([s["fight_prob"] for s in model_segments])
                overall_nonfight_prob = np.mean([s["nonfight_prob"] for s in model_segments])
            else:
                overall_confidence = 1.0
                overall_fight_prob = 0.0
                overall_nonfight_prob = 1.0

        push({"type": "progress", "step": "done", "msg": "Analysis complete!", "pct": 100})

        push({
            "type": "result",
            "label": overall_label,
            "confidence": round(float(overall_confidence), 4),
            "fight_prob": round(float(overall_fight_prob), 4),
            "nonfight_prob": round(float(overall_nonfight_prob), 4),
            "use_gate": use_gate,
            "gate_triggered": not all(not s["gate_triggered"] for s in segment_results),
            "energy": None,  # per-segment energies are in segment_results
            "threshold": MOTION_GATE_THRESHOLD,
            "num_segments": num_segments,
            "segments_skipped": segments_skipped,
            "segments_processed": num_segments - segments_skipped,
            "any_fight": any_fight,
            "timing": {
                "total_ms": round(t_total, 1),
                "gate_ms": round(total_gate_ms, 1),
                "optflow_ms": round(total_optflow_ms, 1),
                "model_ms": round(total_model_ms, 1),
                "resize_ms": 0,
            },
            "video_info": {
                "frames": total_frames,
                "fps": round(fps, 1),
                "duration_s": round(duration_video, 2),
            },
            "segments": segment_results,
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        push({"type": "error", "msg": str(exc)})
    finally:
        stop_cpu.set()
        push({"type": "done"})


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    f = request.files["video"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    job_id = f"job_{int(time.time() * 1000)}"
    save_path = os.path.join(UPLOAD_FOLDER, f"{job_id}{ext}")
    f.save(save_path)
    return jsonify({"job_id": job_id, "path": save_path})


@app.route("/api/analyze/<job_id>")
def analyze(job_id: str):
    video_path = request.args.get("path")
    use_gate = request.args.get("gate", "false").lower() == "true"

    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 404

    q: queue.Queue = queue.Queue()
    _sse_queues[job_id] = q

    thread = threading.Thread(
        target=run_inference, args=(job_id, video_path, use_gate), daemon=True
    )
    thread.start()

    def generate():
        while True:
            try:
                event = q.get(timeout=30)
            except queue.Empty:
                yield "data: {\"type\":\"heartbeat\"}\n\n"
                continue
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") == "done":
                break
        _sse_queues.pop(job_id, None)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/model_status")
def model_status():
    return jsonify({"loaded": _model is not None, "path": MODEL_PATH,
                    "exists": os.path.exists(MODEL_PATH)})


if __name__ == "__main__":
    print("=" * 60)
    print("  Violence Detection Dashboard (v2)")
    print("  - Process-level CPU tracking")
    print("  - 5-second video segmentation")
    print("  http://127.0.0.1:5000")
    print("=" * 60)
    print("Pre-loading model...")
    get_model()
    print("Model ready. Starting server...")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
