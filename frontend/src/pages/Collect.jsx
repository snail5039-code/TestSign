import { useEffect, useMemo, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

// 수집 스펙: 1초 / T=60
const CAPTURE_MS = 1000;
const TARGET_FRAMES = 60;

const ZERO75 = Array(25 * 3).fill(0);
const ZERO63 = Array(21 * 3).fill(0);
const ZERO210 = Array(70 * 3).fill(0);

const MIRROR = false;

// ✅ 한국어 라벨 맵 저장 키
const KO_MAP_LS_KEY = "label_ko_map_v1";

// overlay connections
const HAND_CONNECTIONS = [
  [0, 1],[1, 2],[2, 3],[3, 4],
  [0, 5],[5, 6],[6, 7],[7, 8],
  [0, 9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
];

const POSE_CONNECTIONS = [
  [11,12],[11,13],[13,15],[12,14],[14,16],
  [11,23],[12,24],[23,24],
  [23,25],[25,27],[24,26],[26,28],
];

// ---------------- UI helpers ----------------
function cn(...xs) {
  return xs.filter(Boolean).join(" ");
}

function Badge({ kind = "gray", children }) {
  const base = "inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold";
  const m = {
    green: "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200",
    yellow: "bg-amber-50 text-amber-700 ring-1 ring-amber-200",
    red: "bg-rose-50 text-rose-700 ring-1 ring-rose-200",
    gray: "bg-slate-100 text-slate-700 ring-1 ring-slate-200",
    blue: "bg-sky-50 text-sky-700 ring-1 ring-sky-200",
  };
  return <span className={cn(base, m[kind] || m.gray)}>{children}</span>;
}

function Button({ variant = "primary", disabled, className = "", ...props }) {
  const base =
    "inline-flex items-center justify-center rounded-xl px-4 py-2 text-sm font-semibold transition " +
    "focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50";
  const styles = {
    primary: "bg-emerald-600 text-white hover:bg-emerald-700 focus:ring-emerald-600",
    dark: "bg-slate-900 text-white hover:bg-slate-800 focus:ring-slate-900",
    light: "bg-white text-slate-900 ring-1 ring-slate-200 hover:bg-slate-50 focus:ring-slate-400",
    ghost: "bg-transparent text-slate-700 hover:bg-slate-100 focus:ring-slate-300",
  };
  return <button className={cn(base, styles[variant] || styles.primary, className)} disabled={disabled} {...props} />;
}

function Input({ className = "", ...props }) {
  return (
    <input
      className={cn(
        "w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm outline-none",
        "focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500",
        className
      )}
      {...props}
    />
  );
}

function Select({ className = "", ...props }) {
  return (
    <select
      className={cn(
        "w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm outline-none",
        "focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500",
        className
      )}
      {...props}
    />
  );
}

// ---------------- naming helpers ----------------
function pad(num, width) {
  const s = String(num);
  return s.length >= width ? s : "0".repeat(width - s.length) + s;
}

/**
 * ⚠️ 파일명 규칙은 여기만 유지하면 됨.
 * 예시: yes_000001.json
 */
function makeFileName(label, idx) {
  const safe = (label || "untitled").trim();
  return `${safe}_${pad(idx, 6)}.json`;
}

// ---------------- data helpers ----------------
function toFlatXYZNormalized(landmarks) {
  const out = [];
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    out.push(lm?.x ?? 0, lm?.y ?? 0, lm?.z ?? 0);
  }
  return out;
}

function resampleByTime(rawFrames, startT, endT, targetN) {
  if (!rawFrames?.length) {
    return Array.from({ length: targetN }, () => ({
      pose: [...ZERO75],
      face: [...ZERO210],
      leftHand: [...ZERO63],
      rightHand: [...ZERO63],
    }));
  }

  const src = rawFrames.slice().sort((a, b) => (a.t ?? 0) - (b.t ?? 0));
  const out = [];
  const span = Math.max(1, endT - startT);
  let j = 0;

  for (let i = 0; i < targetN; i++) {
    const targetT = startT + (span * i) / (targetN - 1);
    while (j + 1 < src.length && src[j + 1].t <= targetT) j++;

    const a = src[j];
    const b = src[Math.min(j + 1, src.length - 1)];
    const choose =
      Math.abs((a.t ?? 0) - targetT) <= Math.abs((b.t ?? 0) - targetT) ? a : b;

    out.push({
      pose: [...(choose.pose || ZERO75)],
      face: [...(choose.face || ZERO210)],
      leftHand: [...(choose.leftHand || ZERO63)],
      rightHand: [...(choose.rightHand || ZERO63)],
    });
  }
  return out;
}

function isNonZeroVec(v, eps = 1e-8) {
  if (!Array.isArray(v) || v.length === 0) return false;
  for (const x of v) {
    if (typeof x === "number" && Math.abs(x) > eps) return true;
  }
  return false;
}

function downloadJSON(filename, obj) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// ✅ labelKoMap localStorage helpers
function loadKoMap() {
  try {
    const raw = localStorage.getItem(KO_MAP_LS_KEY);
    if (!raw) return {};
    const obj = JSON.parse(raw);
    return obj && typeof obj === "object" ? obj : {};
  } catch {
    return {};
  }
}

function saveKoMap(map) {
  try {
    localStorage.setItem(KO_MAP_LS_KEY, JSON.stringify(map));
  } catch {}
}

// ---------------- overlay mapping (오프셋 해결) ----------------
function getFitBox(video, w, h, fit = "contain") {
  const vw = video.videoWidth || 640;
  const vh = video.videoHeight || 480;

  const scale =
    fit === "cover" ? Math.max(w / vw, h / vh) : Math.min(w / vw, h / vh);

  const dw = vw * scale;
  const dh = vh * scale;

  const ox = (w - dw) / 2;
  const oy = (h - dh) / 2;

  return { ox, oy, dw, dh };
}

function normToPx(lm, box, mirror) {
  const nx = lm?.x ?? 0;
  const ny = lm?.y ?? 0;
  const x = box.ox + (mirror ? 1 - nx : nx) * box.dw;
  const y = box.oy + ny * box.dh;
  return { x, y };
}

function drawPoints(ctx, landmarks, box, { mirror = false, r = 3 } = {}) {
  for (const lm of landmarks) {
    const { x, y } = normToPx(lm, box, mirror);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawConnections(ctx, landmarks, connections, box, { mirror = false } = {}) {
  for (const [a, b] of connections) {
    const lma = landmarks[a];
    const lmb = landmarks[b];
    if (!lma || !lmb) continue;
    const p1 = normToPx(lma, box, mirror);
    const p2 = normToPx(lmb, box, mirror);
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  }
}

export default function Collect() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const rafRef = useRef(null);

  const handRef = useRef(null);
  const poseRef = useRef(null);

  const [ready, setReady] = useState(false);
  const [camOn, setCamOn] = useState(false);

  // ✅ 통합 모드: Any / 1H / 2H
  const [collectMode, setCollectMode] = useState("any"); // "any" | "1h" | "2h"

  // label & counter (localStorage: 라벨별 카운터)
  const [label, setLabel] = useState("yes");

  // ✅ 한국어 라벨
  const [labelKo, setLabelKo] = useState(() => {
    const m = loadKoMap();
    return (m["yes"] || "").trim();
  });

  // ✅ ko map in-memory
  const koMapRef = useRef(loadKoMap());

  const counterKey = useMemo(
    () => `collect_counter_${(label || "untitled").trim()}`,
    [label]
  );

  const [nextIdx, setNextIdx] = useState(() => {
    try {
      return Number(localStorage.getItem("collect_counter_yes") || 1);
    } catch {
      return 1;
    }
  });

  // status flags
  const armedRef = useRef(false);
  const recordingRef = useRef(false);

  const [armedUI, setArmedUI] = useState(false);
  const [recordingUI, setRecordingUI] = useState(false);
  const [capturedUI, setCapturedUI] = useState(false);

  // buffers
  const rawFramesRef = useRef([]);
  const finalFramesRef = useRef([]);
  const captureStartRef = useRef(0);
  const captureEndRef = useRef(0);

  // hold-last
  const lastLeftRef = useRef([...ZERO63]);
  const lastRightRef = useRef([...ZERO63]);
  const missLeftRef = useRef(0);
  const missRightRef = useRef(0);

  // UI numbers
  const [msLeft, setMsLeft] = useState(0);
  const [rawCount, setRawCount] = useState(0);
  const [finalCount, setFinalCount] = useState(0);

  // 관측값 (디버깅/2H 보장)
  const maxHandsSeenRef = useRef(0);
  const bothHandsFramesRef = useRef(0);

  const fileNamePreview = useMemo(() => makeFileName(label, nextIdx), [label, nextIdx]);

  // label 바뀌면 counter도 그 라벨 기준으로 로드 + labelKo 자동 세팅(있으면)
  useEffect(() => {
    try {
      const v = Number(localStorage.getItem(counterKey) || 1);
      setNextIdx(Number.isFinite(v) && v > 0 ? v : 1);
    } catch {
      setNextIdx(1);
    }

    // ✅ 라벨 변경 시 labelKo 자동 채움
    const m = koMapRef.current || {};
    const ko = (m[(label || "").trim()] || "").trim();
    setLabelKo(ko);
  }, [counterKey, label]);

  // labelKo 입력 변경 시 map 저장(자동)
  useEffect(() => {
    const en = (label || "").trim();
    if (!en) return;

    const ko = (labelKo || "").trim();
    const m = koMapRef.current || {};
    if (ko) m[en] = ko;
    else delete m[en];

    koMapRef.current = { ...m };
    saveKoMap(koMapRef.current);
  }, [labelKo, label]);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      const fileset = await FilesetResolver.forVisionTasks(WASM_URL);

      const hand = await HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: HAND_MODEL },
        runningMode: "VIDEO",
        numHands: 2,
      });

      const pose = await PoseLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: POSE_MODEL },
        runningMode: "VIDEO",
      });

      if (cancelled) return;
      handRef.current = hand;
      poseRef.current = pose;
      setReady(true);
    }

    init();

    return () => {
      cancelled = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      handRef.current?.close?.();
      poseRef.current?.close?.();
    };
  }, []);

  function drawOverlay(handRes, poseRes) {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext?.("2d");
    if (!video || !canvas || !ctx) return;

    const rect = video.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    const dpr = window.devicePixelRatio || 1;
    const cw = Math.round(w * dpr);
    const ch = Math.round(h * dpr);

    if (canvas.width !== cw) canvas.width = cw;
    if (canvas.height !== ch) canvas.height = ch;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    ctx.save();
    ctx.lineWidth = 2;

    const box = getFitBox(video, w, h, "contain");

    const poseLm = poseRes?.landmarks?.[0];
    if (poseLm?.length) {
      const p = poseLm.slice(0, 25);
      ctx.strokeStyle = "rgba(0,255,255,0.70)";
      drawConnections(ctx, p, POSE_CONNECTIONS, box, { mirror: MIRROR });
      ctx.fillStyle = "rgba(0,255,255,0.95)";
      drawPoints(ctx, p, box, { mirror: MIRROR, r: 3 });
    }

    const list = handRes?.landmarks || [];
    for (const lm of list) {
      ctx.strokeStyle = "rgba(0,255,0,0.70)";
      drawConnections(ctx, lm, HAND_CONNECTIONS, box, { mirror: MIRROR });
      ctx.fillStyle = "rgba(0,255,0,0.95)";
      drawPoints(ctx, lm, box, { mirror: MIRROR, r: 3 });
    }

    ctx.restore();
  }

  function shouldStartCapture(handCount, leftFinal, rightFinal) {
    const lh = isNonZeroVec(leftFinal);
    const rh = isNonZeroVec(rightFinal);

    if (collectMode === "1h") return handCount >= 1 && (lh || rh);
    if (collectMode === "2h") return handCount >= 2 && lh && rh;
    return handCount >= 1 && (lh || rh); // any
  }

  async function startCam() {
    if (!ready || camOn) return;

    const video = videoRef.current;
    if (!video) return;

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();

    rawFramesRef.current = [];
    finalFramesRef.current = [];
    setRawCount(0);
    setFinalCount(0);

    armedRef.current = false;
    recordingRef.current = false;
    setArmedUI(false);
    setRecordingUI(false);
    setCapturedUI(false);

    lastLeftRef.current = [...ZERO63];
    lastRightRef.current = [...ZERO63];
    missLeftRef.current = 0;
    missRightRef.current = 0;

    maxHandsSeenRef.current = 0;
    bothHandsFramesRef.current = 0;

    setCamOn(true);

    const loop = () => {
      rafRef.current = requestAnimationFrame(loop);

      const hand = handRef.current;
      const pose = poseRef.current;
      if (!hand || !pose) return;
      if (video.readyState < 2) return;

      const now = performance.now();

      const handRes = hand.detectForVideo(video, now);
      const poseRes = pose.detectForVideo(video, now);

      drawOverlay(handRes, poseRes);

      const handCount = handRes?.landmarks?.length ? handRes.landmarks.length : 0;
      if (handCount > maxHandsSeenRef.current) maxHandsSeenRef.current = handCount;

      let left = null;
      let right = null;

      if (handCount) {
        for (let i = 0; i < handRes.landmarks.length; i++) {
          const lm = handRes.landmarks[i];
          const handed = handRes.handednesses?.[i]?.[0]?.categoryName;
          const flat = toFlatXYZNormalized(lm);

          if (handed === "Left") left = flat;
          else if (handed === "Right") right = flat;
          else {
            if (!left) left = flat;
            else if (!right) right = flat;
          }
        }
      }

      // hold-last
      if (left) { lastLeftRef.current = left; missLeftRef.current = 0; }
      else missLeftRef.current += 1;

      if (right) { lastRightRef.current = right; missRightRef.current = 0; }
      else missRightRef.current += 1;

      const leftFinal = missLeftRef.current <= 5 ? lastLeftRef.current : ZERO63;
      const rightFinal = missRightRef.current <= 5 ? lastRightRef.current : ZERO63;

      if (isNonZeroVec(leftFinal) && isNonZeroVec(rightFinal)) bothHandsFramesRef.current += 1;

      // pose
      let poseFlat = ZERO75;
      if (poseRes?.landmarks?.[0]?.length) {
        const p = poseRes.landmarks[0].slice(0, 25);
        poseFlat = toFlatXYZNormalized(p);
      }
      const faceFlat = ZERO210;

      // armed -> start
      if (armedRef.current && !recordingRef.current) {
        if (shouldStartCapture(handCount, leftFinal, rightFinal)) {
          rawFramesRef.current = [];
          finalFramesRef.current = [];
          setRawCount(0);
          setFinalCount(0);

          captureStartRef.current = now;
          captureEndRef.current = now + CAPTURE_MS;

          recordingRef.current = true;
          armedRef.current = false;

          setArmedUI(false);
          setRecordingUI(true);
          setCapturedUI(false);
        }
      }

      // recording: raw 저장
      if (recordingRef.current) {
        const endT = captureEndRef.current;
        setMsLeft(Math.ceil(Math.max(0, endT - now)));

        rawFramesRef.current.push({
          t: now,
          pose: [...poseFlat],
          face: [...faceFlat],
          leftHand: [...leftFinal],
          rightHand: [...rightFinal],
        });
        setRawCount(rawFramesRef.current.length);

        if (now >= endT) {
          recordingRef.current = false;
          setRecordingUI(false);
          setMsLeft(0);

          const sampled = resampleByTime(
            rawFramesRef.current,
            captureStartRef.current,
            captureEndRef.current,
            TARGET_FRAMES
          );

          finalFramesRef.current = sampled;
          setFinalCount(sampled.length);
          setCapturedUI(true);
        }
      }
    };

    rafRef.current = requestAnimationFrame(loop);
  }

  function stopCam() {
    setCamOn(false);

    armedRef.current = false;
    recordingRef.current = false;
    setArmedUI(false);
    setRecordingUI(false);
    setCapturedUI(false);

    if (rafRef.current) cancelAnimationFrame(rafRef.current);

    const video = videoRef.current;
    if (video?.srcObject) {
      const tracks = video.srcObject.getTracks();
      tracks.forEach((t) => t.stop());
      video.srcObject = null;
    }

    const canvas = canvasRef.current;
    const ctx = canvas?.getContext?.("2d");
    if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function record() {
    if (!camOn) return;
    if (armedRef.current || recordingRef.current) return;

    rawFramesRef.current = [];
    finalFramesRef.current = [];
    setRawCount(0);
    setFinalCount(0);
    setMsLeft(0);

    maxHandsSeenRef.current = 0;
    bothHandsFramesRef.current = 0;

    armedRef.current = true;
    recordingRef.current = false;
    setArmedUI(true);
    setRecordingUI(false);
    setCapturedUI(false);
  }

  function saveJSON() {
    if (!capturedUI) return;
    const frames = finalFramesRef.current;
    if (!frames || frames.length !== TARGET_FRAMES) return;

    const en = (label || "untitled").trim();
    const ko = (labelKo || "").trim();

    // ✅ 여기서 JSON에 labelKo 포함
    const out = {
      label: en,
      labelKo: ko, // <- 추가됨
      frames: frames.map((f) => ({
        pose: f.pose,
        face: f.face,
        leftHand: f.leftHand,
        rightHand: f.rightHand,
      })),
      meta: {
        collect_mode: collectMode,
        capture_ms: CAPTURE_MS,
        target_frames: TARGET_FRAMES,
        mirror: MIRROR,
        created_at: new Date().toISOString(),
        max_hands_seen: maxHandsSeenRef.current,
        both_hands_frames: bothHandsFramesRef.current,
      },
    };

    const filename = makeFileName(en, nextIdx);
    downloadJSON(filename, out);

    const next = nextIdx + 1;
    setNextIdx(next);
    try { localStorage.setItem(counterKey, String(next)); } catch {}
  }

  // ✅ label_ko_map.json 다운로드(학습에 바로 씀)
  function exportLabelKoMap() {
    const m = koMapRef.current || {};
    const out = {
      label_ko_map: m,
      updated_at: new Date().toISOString(),
    };
    downloadJSON("label_ko_map.json", out);
  }

  const progressPct =
    recordingUI && captureStartRef.current > 0
      ? Math.min(100, Math.round(((CAPTURE_MS - msLeft) / CAPTURE_MS) * 100))
      : capturedUI
      ? 100
      : 0;

  const status = !ready
    ? { text: "loading…", kind: "yellow" }
    : !camOn
    ? { text: "cam off", kind: "red" }
    : recordingUI
    ? { text: "recording", kind: "green" }
    : armedUI
    ? { text: "armed", kind: "yellow" }
    : capturedUI
    ? { text: "captured", kind: "green" }
    : { text: "ready", kind: "gray" };

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="sticky top-0 z-10 border-b bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <div>
            <div className="text-base font-extrabold tracking-tight text-slate-900">
              Collect (1H/2H 통합)
            </div>
            <div className="text-xs text-slate-500">
              1s capture / T=60 / 손 감지 시작 / overlay (offset fixed) / labelKo 포함
            </div>
          </div>
          <Badge kind={status.kind}>{status.text}</Badge>
        </div>
      </div>

      <div className="mx-auto max-w-6xl px-4 py-4">
        <div className="grid gap-4 lg:grid-cols-2">
          {/* Left: video */}
          <div className="overflow-hidden rounded-2xl border bg-white shadow-sm">
            <div className="relative">
              <video
                ref={videoRef}
                playsInline
                muted
                className={cn(
                  "aspect-video w-full bg-black object-contain",
                  MIRROR ? "scale-x-[-1]" : ""
                )}
              />
              <canvas
                ref={canvasRef}
                className={cn(
                  "absolute inset-0 h-full w-full pointer-events-none",
                  MIRROR ? "scale-x-[-1]" : ""
                )}
              />
            </div>

            <div className="p-4">
              <div className="flex flex-wrap gap-2">
                <Button variant="primary" disabled={!ready || camOn} onClick={startCam}>
                  Start Cam
                </Button>
                <Button variant="dark" disabled={!camOn} onClick={stopCam}>
                  Stop Cam
                </Button>
                <Button variant="light" disabled={!camOn || armedUI || recordingUI} onClick={record}>
                  Record (wait hand)
                </Button>
                <Button variant="primary" disabled={!capturedUI || armedUI || recordingUI} onClick={saveJSON}>
                  Save JSON
                </Button>
                <Button variant="ghost" disabled={false} onClick={exportLabelKoMap}>
                  Export label_ko_map.json
                </Button>
              </div>

              <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-slate-100">
                <div className="h-full bg-emerald-500 transition-all" style={{ width: `${progressPct}%` }} />
              </div>

              <div className="mt-3 grid gap-1 text-sm text-slate-700">
                <div className="text-slate-500">
                  rawFrames=<span className="font-semibold text-slate-900">{rawCount}</span>{" "}
                  | finalFrames=<span className="font-semibold text-slate-900">{finalCount}/{TARGET_FRAMES}</span>{" "}
                  | msLeft=<span className="font-semibold text-slate-900">{recordingUI ? msLeft : 0}</span>
                </div>
                <div className="text-xs text-slate-400">
                  maxHandsSeen={maxHandsSeenRef.current} | bothHandsFrames={bothHandsFramesRef.current}
                </div>
              </div>
            </div>
          </div>

          {/* Right: meta */}
          <div className="overflow-hidden rounded-2xl border bg-white shadow-sm">
            <div className="border-b bg-white px-4 py-3">
              <div className="text-sm font-bold text-slate-900">메타 / 파일 저장</div>
              <div className="text-xs text-slate-500">
                라벨(영) + 라벨(한) 입력 → 모드 선택 → Save JSON 하면 JSON에 labelKo까지 저장됨
              </div>
            </div>

            <div className="p-4">
              <div className="grid gap-4">
                <div>
                  <label className="text-xs font-semibold text-slate-600">라벨(영어, 폴더명)</label>
                  <Input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="예: yes" />
                  <div className="mt-2 text-xs text-slate-500">
                    이 값이 학습 라벨(클래스) 기준입니다. (dataset/yes/*.json)
                  </div>
                </div>

                <div>
                  <label className="text-xs font-semibold text-slate-600">라벨(한국어, 표시용)</label>
                  <Input value={labelKo} onChange={(e) => setLabelKo(e.target.value)} placeholder="예: 예(yes)" />
                  <div className="mt-2 text-xs text-slate-500">
                    입력하면 자동으로 localStorage에 {`{영어라벨: 한국어라벨}`} 형태로 저장됩니다.
                  </div>
                </div>

                <div>
                  <label className="text-xs font-semibold text-slate-600">수집 모드</label>
                  <Select value={collectMode} onChange={(e) => setCollectMode(e.target.value)}>
                    <option value="any">Any (손 1개라도 감지되면 시작)</option>
                    <option value="1h">1H Only (한손 데이터 위주)</option>
                    <option value="2h">2H Only (양손 모두 잡혀야 시작)</option>
                  </Select>
                  <div className="mt-2 text-xs text-slate-500">
                    2H는 시작 조건을 엄격하게 해서 “두손 단어”가 섞이지 않게 합니다.
                  </div>
                </div>

                <div className="rounded-xl border bg-slate-50 p-3">
                  <div className="text-xs font-semibold text-slate-600">저장 파일명(프리뷰)</div>
                  <div className="mt-1 font-mono text-sm text-slate-900">{fileNamePreview}</div>
                </div>

                <div className="rounded-xl border bg-slate-50 p-3">
                  <div className="text-xs font-semibold text-slate-600">다음 저장 번호</div>
                  <div className="mt-1 text-sm text-slate-900">
                    {label.trim() || "untitled"} : <span className="font-mono">{pad(nextIdx, 6)}</span>
                  </div>
                  <div className="mt-2 text-xs text-slate-500">(브라우저 localStorage에 저장됨)</div>
                </div>

                <div className="text-xs text-slate-500 leading-relaxed">
                  Record를 누르면 “조건 만족 시점부터” 1초 캡처 후 60프레임으로 리샘플합니다.
                  <br />
                  Save JSON을 누르면 현재 캡처된 60프레임을 JSON으로 다운로드합니다.
                  <br />
                  Export label_ko_map.json 버튼으로 전체 한국어 매핑도 같이 뽑을 수 있습니다.
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 text-xs text-slate-400">
          이 페이지는 통합 수집입니다. “한손 전용”은 <span className="font-mono">Collect1H.jsx</span>를 별도로 사용하세요.
        </div>
      </div>
    </div>
  );
}
