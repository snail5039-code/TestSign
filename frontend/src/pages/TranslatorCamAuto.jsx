import { useEffect, useMemo, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

// 1초 / T=60
const CAPTURE_MS = 1000;
const TARGET_FRAMES = 60;

// FastAPI 직통
const API_URL = "http://127.0.0.1:8000/predict/auto";
// Spring 프록시 쓰면: "http://127.0.0.1:8080/api/translate/auto"

const ZERO75 = Array(25 * 3).fill(0);
const ZERO63 = Array(21 * 3).fill(0);
const ZERO210 = Array(70 * 3).fill(0);

const MIRROR = false;

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

// ---------------- data helpers ----------------
function toFlatXYZNormalized(landmarks) {
  const out = [];
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    out.push(lm?.x ?? 0, lm?.y ?? 0, lm?.z ?? 0);
  }
  return out;
}

function deepCloneFrames(frames) {
  return frames.map((f) => ({
    pose: Array.isArray(f.pose) ? [...f.pose] : [...ZERO75],
    face: Array.isArray(f.face) ? [...f.face] : [...ZERO210],
    leftHand: Array.isArray(f.leftHand) ? [...f.leftHand] : [...ZERO63],
    rightHand: Array.isArray(f.rightHand) ? [...f.rightHand] : [...ZERO63],
  }));
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

function computeBothRatio(frames) {
  if (!frames?.length) return { bothRatio: 0, anyRatio: 0 };
  let both = 0, anyh = 0;
  for (const fr of frames) {
    const lh = isNonZeroVec(fr?.leftHand);
    const rh = isNonZeroVec(fr?.rightHand);
    if (lh || rh) anyh++;
    if (lh && rh) both++;
  }
  return { bothRatio: both / frames.length, anyRatio: anyh / frames.length };
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

export default function TranslatorCamAuto() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const rafRef = useRef(null);

  const handRef = useRef(null);
  const poseRef = useRef(null);

  const [ready, setReady] = useState(false);
  const [camOn, setCamOn] = useState(false);

  const armedRef = useRef(false);
  const recordingRef = useRef(false);

  const [armedUI, setArmedUI] = useState(false);
  const [recordingUI, setRecordingUI] = useState(false);
  const [capturedUI, setCapturedUI] = useState(false);

  const rawFramesRef = useRef([]);
  const finalFramesRef = useRef([]);
  const captureStartRef = useRef(0);
  const captureEndRef = useRef(0);

  const maxHandsSeenRef = useRef(0);

  // hold-last
  const lastLeftRef = useRef([...ZERO63]);
  const lastRightRef = useRef([...ZERO63]);
  const missLeftRef = useRef(0);
  const missRightRef = useRef(0);

  // UI
  const [msLeft, setMsLeft] = useState(0);
  const [rawCount, setRawCount] = useState(0);
  const [finalCount, setFinalCount] = useState(0);

  // 결과
  const [mode, setMode] = useState("-");
  const [label, setLabel] = useState("-");
  const [confidence, setConfidence] = useState(0);
  const [top5, setTop5] = useState([]);

  const inFlightRef = useRef(false);

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

  const progressPct =
    recordingUI && captureStartRef.current > 0
      ? Math.min(100, Math.round(((CAPTURE_MS - msLeft) / CAPTURE_MS) * 100))
      : capturedUI
      ? 100
      : 0;

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

    maxHandsSeenRef.current = 0;

    lastLeftRef.current = [...ZERO63];
    lastRightRef.current = [...ZERO63];
    missLeftRef.current = 0;
    missRightRef.current = 0;

    setMode("-");
    setLabel("-");
    setConfidence(0);
    setTop5([]);

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

      // pose
      let poseFlat = ZERO75;
      if (poseRes?.landmarks?.[0]?.length) {
        const p = poseRes.landmarks[0].slice(0, 25);
        poseFlat = toFlatXYZNormalized(p);
      }
      const faceFlat = ZERO210;

      // armed: 손 감지되면 시작
      if (armedRef.current && !recordingRef.current) {
        const anyHandNow = handCount > 0 && (isNonZeroVec(leftFinal) || isNonZeroVec(rightFinal));
        if (anyHandNow) {
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

      // recording
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

    armedRef.current = true;
    recordingRef.current = false;
    setArmedUI(true);
    setRecordingUI(false);
    setCapturedUI(false);

    setMode("-");
    setLabel("-");
    setConfidence(0);
    setTop5([]);
  }

  async function sendTranslate() {
    if (!capturedUI) return;
    if (inFlightRef.current) return;

    const frames = deepCloneFrames(finalFramesRef.current);
    if (frames.length !== TARGET_FRAMES) return;

    const ratios = computeBothRatio(frames);
    const hint = {
      maxHandsSeen: maxHandsSeenRef.current,
      bothRatio: ratios.bothRatio,
    };

    inFlightRef.current = true;
    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frames, hint }),
      });

      if (!res.ok) {
        const t = await res.text();
        throw new Error(`translate failed ${res.status}: ${t}`);
      }

      const data = await res.json();
      setMode(data?.mode ?? "-");
      setLabel(data?.label ?? "-");
      setConfidence(Number(data?.confidence ?? 0));

      const t5 = Array.isArray(data?.top5)
        ? data.top5.map((obj) => {
            const k = Object.keys(obj || {})[0];
            const v = k ? obj[k] : null;
            return { label: k ?? "-", score: typeof v === "number" ? v : null };
          })
        : [];
      setTop5(t5);
    } catch (e) {
      console.error(e);
    } finally {
      inFlightRef.current = false;
    }
  }

  const top5Rows = useMemo(() => {
    if (!Array.isArray(top5) || top5.length === 0) return [];
    return top5.map((x, idx) => ({
      idx,
      label: x?.label ?? "-",
      score: typeof x?.score === "number" ? x.score : null,
    }));
  }, [top5]);

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="sticky top-0 z-10 border-b bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <div>
            <div className="text-base font-extrabold tracking-tight text-slate-900">
              Translator (Auto)
            </div>
            <div className="text-xs text-slate-500">
              any-hand start → 1s capture → T=60 → /predict/auto (overlay fixed)
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge kind="blue">API</Badge>
            <span className="hidden sm:block text-xs font-mono text-slate-600">{API_URL}</span>
            <Badge kind={status.kind}>{status.text}</Badge>
          </div>
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
                <Button
                  variant="primary"
                  disabled={!capturedUI || armedUI || recordingUI}
                  onClick={sendTranslate}
                >
                  Send (Translate)
                </Button>
              </div>

              <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-slate-100">
                <div className="h-full bg-emerald-500 transition-all" style={{ width: `${progressPct}%` }} />
              </div>

              <div className="mt-3 grid gap-1 text-sm text-slate-700">
                <div className="text-slate-500">
                  rawFrames=<span className="font-semibold text-slate-900">{rawCount}</span>{" "}
                  | finalFrames=<span className="font-semibold text-slate-900">{finalCount}/{TARGET_FRAMES}</span>{" "}
                  | msLeft=<span className="font-semibold text-slate-900">{recordingUI ? msLeft : 0}</span>{" "}
                  | maxHandsSeen=<span className="font-semibold text-slate-900">{maxHandsSeenRef.current}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Right: result */}
          <div className="overflow-hidden rounded-2xl border bg-white shadow-sm">
            <div className="border-b bg-white px-4 py-3">
              <div className="text-sm font-bold text-slate-900">번역 결과</div>
              <div className="text-xs text-slate-500">
                mode는 서버 auto route가 선택한 모델(1h/2h)입니다.
              </div>
            </div>

            <div className="p-4">
              <div className="grid gap-4">
                <div className="grid grid-cols-3 gap-3">
                  <div className="rounded-xl border bg-slate-50 p-3">
                    <div className="text-xs font-semibold text-slate-600">mode</div>
                    <div className="mt-1 text-lg font-extrabold text-slate-900">{mode}</div>
                  </div>
                  <div className="rounded-xl border bg-slate-50 p-3 col-span-2">
                    <div className="text-xs font-semibold text-slate-600">label</div>
                    <div className="mt-1 text-2xl font-extrabold text-slate-900">{label}</div>
                  </div>
                </div>

                <div className="rounded-xl border bg-slate-50 p-3">
                  <div className="text-xs font-semibold text-slate-600">confidence</div>
                  <div className="mt-1 font-mono text-sm text-slate-900">
                    {Number.isFinite(confidence) ? confidence.toFixed(6) : "0.000000"}
                  </div>
                </div>

                <div className="rounded-xl border bg-slate-50 p-3">
                  <div className="text-xs font-semibold text-slate-600">top5</div>
                  {top5Rows.length === 0 ? (
                    <div className="mt-2 text-sm text-slate-500">-</div>
                  ) : (
                    <ol className="mt-2 space-y-1 text-sm text-slate-800">
                      {top5Rows.map((x) => (
                        <li key={x.idx} className="flex items-center justify-between">
                          <span className="font-semibold">{x.label}</span>
                          <span className="font-mono text-xs text-slate-500">
                            {typeof x.score === "number" ? x.score.toFixed(6) : ""}
                          </span>
                        </li>
                      ))}
                    </ol>
                  )}
                </div>

                <div className="text-xs text-slate-400 leading-relaxed">
                  Record를 누르면 손이 감지되는 순간부터 1초 캡처 후 60프레임으로 리샘플합니다.
                  <br />
                  Send를 누르면 서버가 bothRatio로 1h/2h 모델을 자동 선택합니다.
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-4 text-xs text-slate-400">
          파일: <span className="font-mono">src/pages/TranslatorCamAuto.jsx</span>
        </div>
      </div>
    </div>
  );
}
