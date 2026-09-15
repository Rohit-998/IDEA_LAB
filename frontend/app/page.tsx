"use client";

import { useRef, useState, useCallback } from "react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";

// ─── Types ────────────────────────────────────────────────────────────────────
interface CpuPoint { ts: number; cpu: number; mem_mb: number; }

interface SegmentResult {
  seg_index: number;
  start_s: number;
  end_s: number;
  frames: number;
  gate_triggered: boolean;
  energy: number | null;
  gate_ms: number;
  optflow_ms: number;
  model_ms: number;
  total_ms: number;
  label: string;
  confidence: number;
  fight_prob: number;
  nonfight_prob: number;
  is_fight: boolean;
}

interface AnalysisResult {
  label: string;
  confidence: number;
  fight_prob: number;
  nonfight_prob: number;
  use_gate: boolean;
  gate_triggered: boolean;
  energy: number | null;
  threshold: number;
  num_segments: number;
  segments_skipped: number;
  segments_processed: number;
  any_fight: boolean;
  timing: {
    total_ms: number;
    gate_ms: number;
    optflow_ms: number;
    model_ms: number;
    resize_ms: number;
  };
  video_info: { frames: number; fps: number; duration_s: number };
  segments: SegmentResult[];
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
const BACKEND = "http://localhost:5000";

function fmtMs(ms: number) {
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`;
  return `${ms.toFixed(0)}ms`;
}
function fmtBytes(b: number) {
  if (b >= 1024 * 1024) return `${(b / 1024 / 1024).toFixed(1)} MB`;
  if (b >= 1024) return `${(b / 1024).toFixed(0)} KB`;
  return `${b} B`;
}
function cpuClass(v: number) {
  if (v >= 70) return "high";
  if (v >= 40) return "med";
  return "good";
}

// ─── Component ────────────────────────────────────────────────────────────────
export default function Dashboard() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [gateEnabled, setGateEnabled] = useState(false);

  // Job state
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentMsg, setCurrentMsg] = useState("");

  // CPU data (process-level)
  const [cpuData, setCpuData] = useState<CpuPoint[]>([]);
  const [currentCpu, setCurrentCpu] = useState(0);
  const [peakCpu, setPeakCpu] = useState(0);
  const [currentMem, setCurrentMem] = useState(0);

  // Segment results (live)
  const [liveSegments, setLiveSegments] = useState<SegmentResult[]>([]);

  // Final results
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Comparison
  const [noGateResult, setNoGateResult] = useState<AnalysisResult | null>(null);
  const [gateResult, setGateResult] = useState<AnalysisResult | null>(null);

  // ── File handling ──────────────────────────────────────────────────────────
  const handleFile = useCallback((f: File) => {
    setFile(f);
    setResult(null);
    setError(null);
    setCpuData([]);
    setProgress(0);
    setCurrentMsg("");
    setPeakCpu(0);
    setLiveSegments([]);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent) => { e.preventDefault(); setDragging(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); },
    [handleFile]
  );

  // ── Analyze ────────────────────────────────────────────────────────────────
  const analyze = useCallback(async () => {
    if (!file) return;
    setRunning(true);
    setResult(null);
    setError(null);
    setCpuData([]);
    setProgress(0);
    setPeakCpu(0);
    setLiveSegments([]);

    try {
      const form = new FormData();
      form.append("video", file);
      setCurrentMsg("Uploading video…");
      const upRes = await fetch(`${BACKEND}/api/upload`, { method: "POST", body: form });
      if (!upRes.ok) throw new Error(`Upload failed: ${upRes.statusText}`);
      const { job_id, path } = await upRes.json();

      const url = `${BACKEND}/api/analyze/${job_id}?path=${encodeURIComponent(path)}&gate=${gateEnabled}`;
      const es = new EventSource(url);

      es.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);

        if (msg.type === "cpu") {
          const point: CpuPoint = { ts: msg.ts, cpu: msg.cpu, mem_mb: msg.mem_mb };
          setCpuData((prev) => [...prev.slice(-120), point]);
          setCurrentCpu(msg.cpu);
          setCurrentMem(msg.mem_mb);
          setPeakCpu((prev) => Math.max(prev, msg.cpu));
        }

        if (msg.type === "progress") {
          setProgress(msg.pct);
          setCurrentMsg(msg.msg);
        }

        if (msg.type === "segment_result") {
          setLiveSegments((prev) => [...prev, msg as SegmentResult]);
        }

        if (msg.type === "result") {
          const r = msg as unknown as AnalysisResult;
          setResult(r);
          if (!r.use_gate) setNoGateResult(r);
          else setGateResult(r);
        }

        if (msg.type === "error") {
          setError(msg.msg);
          es.close();
          setRunning(false);
        }

        if (msg.type === "done") {
          es.close();
          setRunning(false);
        }
      };

      es.onerror = () => {
        setError("Connection to backend lost. Is Flask running on port 5000?");
        es.close();
        setRunning(false);
      };
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      setRunning(false);
    }
  }, [file, gateEnabled]);

  // ─── UI ────────────────────────────────────────────────────────────────────
  return (
    <div className="app-wrapper">
      {/* Header */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <div className="logo-icon">🛡️</div>
            <div>
              <div className="logo-text">ViolenceGuard</div>
              <div className="logo-sub">Motion-Gated Cascade · RWF-2000</div>
            </div>
          </div>
          <div className="header-badge">
            <div className="pulse-dot" />
            Flow-Gated Network · 87% Accuracy
          </div>
        </div>
      </header>

      {/* Hero */}
      <div className="page-hero">
        <h1 className="page-hero-title">
          CPU Resource Comparison:<br />
          <span className="page-hero-accent">Always-On vs. Motion-Gated Cascade</span>
        </h1>
        <p className="page-hero-desc">
          Upload a video, toggle the cheap first-check gate, and watch real-time process-level CPU usage.
          Videos are split into 5-second segments for accurate predictions.
        </p>
      </div>

      {/* Main grid */}
      <div className="main-grid">
        {/* ── LEFT COLUMN ── */}
        <div className="left-col">
          {/* Upload card */}
          <div className="card">
            <div className="card-title"><span>📤</span> Video Input</div>
            <div
              className={`upload-zone${dragging ? " drag-active" : ""}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
            >
              <div className="upload-icon">
                <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.8" viewBox="0 0 24 24">
                  <path d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </div>
              {file ? (
                <div className="upload-selected">
                  <span style={{ color: "var(--success)" }}>✓</span>
                  <div style={{ flex: 1, overflow: "hidden" }}>
                    <div className="upload-selected-name">{file.name}</div>
                    <div className="upload-selected-size">{fmtBytes(file.size)}</div>
                  </div>
                  <span style={{ fontSize: 11, color: "var(--accent)", cursor: "pointer" }}>change</span>
                </div>
              ) : (
                <>
                  <div className="upload-label">Drop your video here</div>
                  <div className="upload-sub">MP4, AVI, MKV, MOV — any length</div>
                </>
              )}
            </div>
            <input ref={fileInputRef} type="file" accept="video/*" style={{ display: "none" }}
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
          </div>

          {/* Gate toggle */}
          <div className={`gate-card${gateEnabled ? " gate-on" : ""}`}>
            <div className="gate-info">
              <div className="gate-label">⚡ Motion-Energy Gate (Stage 1)</div>
              <div className="gate-desc">
                {gateEnabled
                  ? "Gate ON — each 5s segment is checked with cheap frame-differencing first. Heavy model only fires if motion exceeds threshold."
                  : "Gate OFF — every 5s segment goes through the full pipeline. This is the always-on baseline."}
              </div>
              <div className="gate-chips">
                <span className="gate-chip">Resolution: <b>56×56</b></span>
                <span className="gate-chip">Segment: <b>5 sec</b></span>
                <span className="gate-chip">Threshold: <b>15.0</b></span>
                <span className="gate-chip">Gate cost: <b>~5ms</b></span>
              </div>
            </div>
            <div
              className={`toggle${gateEnabled ? " on" : ""}`}
              onClick={() => !running && setGateEnabled((v) => !v)}
            >
              <div className="toggle-thumb" />
            </div>
          </div>

          {/* Analyze button */}
          <button className="btn-analyze" disabled={!file || running} onClick={analyze}>
            {running ? (
              <><div className="spinner" />{gateEnabled ? "Running cascade pipeline…" : "Running full pipeline…"}</>
            ) : (
              <><span>▶</span>{gateEnabled ? "Analyze with Gate" : "Analyze (Always-On)"}</>
            )}
          </button>

          {error && (
            <div style={{ background: "rgba(255,77,109,0.1)", border: "1px solid rgba(255,77,109,0.35)", borderRadius: 12, padding: "14px 18px", color: "var(--danger)", fontSize: 13 }}>
              ⚠️ {error}
            </div>
          )}

          {/* Segment Timeline */}
          {(running || result) && liveSegments.length > 0 && (
            <div className="card">
              <div className="card-title"><span>🎞️</span> Segment Timeline</div>
              <div style={{ fontSize: 12, color: "var(--text-secondary)", marginBottom: 12 }}>
                {currentMsg}
              </div>
              <div className="progress-bar-track">
                <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                {liveSegments.map((seg) => {
                  const isFight = seg.is_fight;
                  const skipped = !seg.gate_triggered;
                  return (
                    <div key={seg.seg_index} className={`step-item ${isFight ? "active" : skipped ? "skip" : "done"}`}
                      style={isFight ? { borderColor: "rgba(255,77,109,0.5)", background: "rgba(255,77,109,0.08)" } : {}}>
                      <div className="step-icon" style={isFight ? { background: "var(--danger)" } : {}}>
                        {skipped ? "⚡" : isFight ? "⚠️" : "✅"}
                      </div>
                      <div style={{ flex: 1 }}>
                        <span className="step-text" style={isFight ? { color: "var(--danger)", fontWeight: 600 } : {}}>
                          Seg {seg.seg_index + 1} [{seg.start_s.toFixed(1)}s – {seg.end_s.toFixed(1)}s]
                        </span>
                        <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>
                          {skipped
                            ? `Skipped (energy: ${seg.energy?.toFixed(1)}) — ${fmtMs(seg.total_ms)}`
                            : `${seg.label} (${(seg.confidence * 100).toFixed(0)}%) — ${fmtMs(seg.total_ms)}`}
                        </div>
                      </div>
                      {isFight && (
                        <span style={{ fontSize: 10, fontWeight: 700, color: "var(--danger)", background: "rgba(255,77,109,0.15)", padding: "2px 8px", borderRadius: 6 }}>
                          FIGHT
                        </span>
                      )}
                      {skipped && (
                        <span style={{ fontSize: 10, fontWeight: 600, color: "var(--success)", background: "rgba(34,211,165,0.12)", padding: "2px 8px", borderRadius: 6 }}>
                          SKIPPED
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* ── RIGHT COLUMN ── */}
        <div className="right-col">
          {/* CPU chart */}
          <div className="card">
            <div className="card-title"><span>📈</span> Process CPU &amp; Memory (Isolated)</div>
            <div className="cpu-stats-row">
              <div className="cpu-stat-box">
                <div className={`cpu-stat-val ${cpuClass(currentCpu)}`}>{currentCpu.toFixed(1)}%</div>
                <div className="cpu-stat-label">Process CPU</div>
              </div>
              <div className="cpu-stat-box">
                <div className={`cpu-stat-val ${cpuClass(peakCpu)}`}>{peakCpu.toFixed(1)}%</div>
                <div className="cpu-stat-label">Peak CPU</div>
              </div>
              <div className="cpu-stat-box">
                <div className="cpu-stat-val good">{currentMem.toFixed(0)} MB</div>
                <div className="cpu-stat-label">Process RAM</div>
              </div>
            </div>
            <div style={{ height: 200 }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={cpuData} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
                  <defs>
                    <linearGradient id="cpuGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#6c63ff" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#6c63ff" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="memGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#22d3a5" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#22d3a5" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="ts" hide />
                  <YAxis domain={[0, 100]} tick={{ fill: "#4a5568", fontSize: 10 }} tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{ background: "#111827", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, fontSize: 12 }}
                    formatter={(v: unknown, name: unknown) => [`${Number(v).toFixed(1)}${name === "cpu" ? "%" : " MB"}`, name === "cpu" ? "Process CPU" : "Process RAM"]}
                    labelFormatter={() => ""}
                  />
                  <Area type="monotone" dataKey="cpu" stroke="#6c63ff" strokeWidth={2} fill="url(#cpuGrad)" dot={false} isAnimationActive={false} />
                  <Area type="monotone" dataKey="mem_mb" stroke="#22d3a5" strokeWidth={1.5} fill="url(#memGrad)" dot={false} isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div style={{ display: "flex", gap: 16, marginTop: 10, fontSize: 11, color: "var(--text-muted)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 10, height: 2, background: "#6c63ff", borderRadius: 1 }} /> Process CPU %
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <div style={{ width: 10, height: 2, background: "#22d3a5", borderRadius: 1 }} /> Process RAM (MB)
              </div>
              {!running && cpuData.length === 0 && (
                <span style={{ marginLeft: "auto" }}>Isolated from background apps ✓</span>
              )}
            </div>
          </div>

          {/* Result card */}
          {result ? (
            <ResultCard result={result} />
          ) : (
            <div className="card">
              <div className="card-title"><span>🎯</span> Prediction Result</div>
              <div className="empty-state">
                <div className="empty-icon">
                  <svg width="28" height="28" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                    <circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3" strokeLinecap="round"/>
                  </svg>
                </div>
                <div style={{ fontSize: 14, color: "var(--text-muted)" }}>Awaiting analysis</div>
                <div style={{ fontSize: 12, color: "var(--text-muted)", opacity: 0.6 }}>Upload a video and click Analyze</div>
              </div>
            </div>
          )}

          {/* Comparison */}
          {noGateResult && gateResult && (
            <ComparisonCard noGate={noGateResult} gate={gateResult} />
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Result card ──────────────────────────────────────────────────────────────
function ResultCard({ result }: { result: AnalysisResult }) {
  const isFight = result.any_fight;
  const allSkipped = result.segments_skipped === result.num_segments;
  const vclass = isFight ? "fight" : allSkipped ? "calm" : "nonfight";
  const { timing } = result;
  const maxBar = Math.max(timing.optflow_ms, timing.model_ms, timing.gate_ms, 1);

  return (
    <div className={`result-card ${vclass}`} style={{ animation: "slideUp 0.5s ease" }}>
      <style>{`@keyframes slideUp { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }`}</style>

      {/* Gate pill */}
      {result.use_gate ? (
        <div className="gate-pill passed">
          ⚡ Cascade: {result.segments_processed} processed / {result.segments_skipped} skipped out of {result.num_segments} segments
        </div>
      ) : (
        <div className="gate-pill no-gate">🔴 Always-On: all {result.num_segments} segments fully processed</div>
      )}

      {/* Verdict */}
      <div className={`result-verdict ${vclass}`}>
        {isFight ? "⚠️ Violence Detected" : allSkipped ? "✅ All Segments Calm" : "✅ Non-Violent"}
      </div>
      <div className="result-sub">
        {result.label} · {(result.confidence * 100).toFixed(1)}% confidence
        · {result.num_segments} segment{result.num_segments > 1 ? "s" : ""}
      </div>

      {/* Confidence */}
      <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 5 }}>Fight probability</div>
      <div className="confidence-bar">
        <div className={`confidence-fill ${isFight ? "fight" : "nonfight"}`}
          style={{ width: `${result.fight_prob * 100}%` }} />
      </div>
      <div className="confidence-labels">
        <span style={{ color: "var(--danger)" }}>{(result.fight_prob * 100).toFixed(1)}% Fight</span>
        <span>{(result.nonfight_prob * 100).toFixed(1)}% NonFight</span>
      </div>

      <div className="glow-divider" />

      {/* Timing */}
      <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 10, textTransform: "uppercase", letterSpacing: "0.08em" }}>
        Total Timing ({result.num_segments} segments)
      </div>
      <div className="timing-grid">
        <div className="timing-item">
          <div className="timing-label">Total time</div>
          <div className={`timing-val ${timing.total_ms < 2000 ? "fast" : "slow"}`}>{fmtMs(timing.total_ms)}</div>
          <div className="timing-bar" style={{ width: "100%", background: "var(--grad-accent)" }} />
        </div>
        <div className="timing-item">
          <div className="timing-label">Optical flow</div>
          <div className={`timing-val ${timing.optflow_ms > 2000 ? "slow" : "fast"}`}>{fmtMs(timing.optflow_ms)}</div>
          <div className="timing-bar" style={{ width: `${(timing.optflow_ms / maxBar) * 100}%`, background: "var(--grad-danger)" }} />
        </div>
        <div className="timing-item">
          <div className="timing-label">Model inference</div>
          <div className="timing-val">{fmtMs(timing.model_ms)}</div>
          <div className="timing-bar" style={{ width: `${(timing.model_ms / maxBar) * 100}%`, background: "var(--grad-accent)" }} />
        </div>
        <div className="timing-item">
          <div className="timing-label">Gate (all segs)</div>
          <div className="timing-val fast">{timing.gate_ms > 0 ? fmtMs(timing.gate_ms) : "—"}</div>
          <div className="timing-bar" style={{ width: `${(timing.gate_ms / maxBar) * 100}%`, background: "var(--grad-success)" }} />
        </div>
      </div>

      <div className="glow-divider" />
      <div style={{ display: "flex", gap: 16, fontSize: 12, color: "var(--text-muted)" }}>
        <span>🎬 {result.video_info.frames} frames</span>
        <span>🎞️ {result.video_info.fps} FPS</span>
        <span>⏱️ {result.video_info.duration_s}s</span>
        <span>📦 {result.num_segments} segments</span>
      </div>
    </div>
  );
}

// ─── Comparison card ──────────────────────────────────────────────────────────
function ComparisonCard({ noGate, gate }: { noGate: AnalysisResult; gate: AnalysisResult }) {
  const speedup = noGate.timing.total_ms / gate.timing.total_ms;
  const saved_ms = noGate.timing.total_ms - gate.timing.total_ms;

  return (
    <div className="card" style={{ borderColor: "rgba(34,211,165,0.3)", boxShadow: "0 0 30px rgba(34,211,165,0.12)" }}>
      <div className="card-title"><span>⚡</span> Head-to-Head Comparison</div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", gap: 12, alignItems: "center", marginBottom: 18 }}>
        <div style={{ background: "var(--bg-surface)", border: "1px solid rgba(255,77,109,0.25)", borderRadius: 12, padding: 14 }}>
          <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6 }}>Always-On</div>
          <div style={{ fontSize: 22, fontWeight: 800, fontFamily: "JetBrains Mono", color: "var(--danger)" }}>
            {fmtMs(noGate.timing.total_ms)}
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
            {noGate.num_segments} segs fully processed
          </div>
        </div>

        <div style={{ width: 36, height: 36, background: "var(--bg-surface)", border: "1px solid var(--border-bright)", borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 700, color: "var(--text-muted)" }}>
          VS
        </div>

        <div style={{ background: "var(--bg-surface)", border: "1px solid rgba(34,211,165,0.25)", borderRadius: 12, padding: 14 }}>
          <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6 }}>With Gate</div>
          <div style={{ fontSize: 22, fontWeight: 800, fontFamily: "JetBrains Mono", color: "var(--success)" }}>
            {fmtMs(gate.timing.total_ms)}
          </div>
          <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
            {gate.segments_skipped} of {gate.num_segments} segs skipped
          </div>
        </div>
      </div>

      <div style={{ textAlign: "center" }}>
        <div className="speedup-badge">🚀 {speedup.toFixed(1)}× faster · saved {fmtMs(saved_ms)}</div>
      </div>

      <div className="glow-divider" />

      <div style={{ fontSize: 11, color: "var(--text-muted)", marginBottom: 10, textTransform: "uppercase", letterSpacing: "0.08em" }}>
        Time breakdown
      </div>
      {[
        { label: "Optical Flow", ng: noGate.timing.optflow_ms, g: gate.timing.optflow_ms },
        { label: "Model Inference", ng: noGate.timing.model_ms, g: gate.timing.model_ms },
        { label: "Gate (Stage 1)", ng: 0, g: gate.timing.gate_ms },
      ].map(({ label, ng, g }) => {
        const max = Math.max(ng, g, 1);
        return (
          <div key={label} style={{ marginBottom: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "var(--text-muted)", marginBottom: 5 }}>
              <span>{label}</span>
              <span>
                {ng > 0 && <span style={{ color: "var(--danger)" }}>{fmtMs(ng)}</span>}
                {ng > 0 && g > 0 && " → "}
                {g > 0 && <span style={{ color: "var(--success)" }}>{fmtMs(g)}</span>}
                {ng > 0 && g === 0 && <span style={{ color: "var(--success)" }}> → skipped</span>}
              </span>
            </div>
            <div style={{ height: 6, background: "var(--bg-surface)", borderRadius: 3, overflow: "hidden", marginBottom: 3 }}>
              <div style={{ height: "100%", width: `${(ng / max) * 100}%`, background: "var(--grad-danger)", borderRadius: 3, opacity: 0.7 }} />
            </div>
            <div style={{ height: 6, background: "var(--bg-surface)", borderRadius: 3, overflow: "hidden" }}>
              <div style={{ height: "100%", width: `${(g / max) * 100}%`, background: "var(--grad-success)", borderRadius: 3, opacity: 0.8 }} />
            </div>
          </div>
        );
      })}

      <div style={{ marginTop: 12, padding: "10px 14px", background: "rgba(34,211,165,0.07)", border: "1px solid rgba(34,211,165,0.2)", borderRadius: 10, fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5 }}>
        ✓ Same detection result · {speedup.toFixed(1)}× compute reduction · Gate cost: {fmtMs(gate.timing.gate_ms)} · {gate.segments_skipped} segments skipped
      </div>
    </div>
  );
}
