import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const API_URL = "http://127.0.0.1:8080/api/translate";

// ===== 새 학습 스펙 =====
const CAPTURE_MS = 1000;
const TARGET_FRAMES = 60;
const RAW_FPS_CAP = 60;
const RAW_INTERVAL_MS = RAW_FPS_CAP > 0 ? 1000 / RAW_FPS_CAP : 0;

const ZERO75 = Array(25 * 3).fill(0);
const ZERO63 = Array(21 * 3).fill(0);
const ZERO210 = Array(70 * 3).fill(0);

const MIRROR = false;

// ===== overlay =====
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

function toFlatXYZNormalized(landmarks) {
  const out = [];
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    const x = lm?.x ?? 0;
    const y = lm?.y ?? 0;
    const z = lm?.z ?? 0;
    out.push(x, y, z);
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

// ===== draw =====
function normToPx(lm, w, h, mirror) {
  const nx = lm?.x ?? 0;
  const ny = lm?.y ?? 0;
  const x = (mirror ? 1 - nx : nx) * w;
  const y = ny * h;
  return { x, y };
}

function drawPoints(ctx, landmarks, w, h, { mirror = false, r = 3 } = {}) {
  for (const lm of landmarks) {
    const { x, y } = normToPx(lm, w, h, mirror);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawConnections(ctx, landmarks, connections, w, h, { mirror = false } = {}) {
  for (const [a, b] of connections) {
    const lma = landmarks[a];
    const lmb = landmarks[b];
    if (!lma || !lmb) continue;
    const p1 = normToPx(lma, w, h, mirror);
    const p2 = normToPx(lmb, w, h, mirror);
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  }
}

export default function TranslatorCam() {
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
  const [captured, setCaptured] = useState(false);

  const [rawCount, setRawCount] = useState(0);
  const [framesCount, setFramesCount] = useState(0);
  const [msLeft, setMsLeft] = useState(0);

  const rawFramesRef = useRef([]);
  const finalFramesRef = useRef([]);

  const captureStartRef = useRef(0);
  const captureEndRef = useRef(0);
  const lastRawPushRef = useRef(0);

  const inFlightRef = useRef(false);

  // output
  const [label, setLabel] = useState("-");
  const [text, setText] = useState("-");
  const [confidence, setConfidence] = useState(0);
  const [top5UI, setTop5UI] = useState([]);

  // hold-last (2손 대응)
  const lastLeftRef = useRef([...ZERO63]);
  const lastRightRef = useRef([...ZERO63]);
  const missLeftRef = useRef(0);
  const missRightRef = useRef(0);

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

  function drawOverlay({ handRes, poseRes, w, h }) {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext?.("2d");
    if (!canvas || !ctx) return;

    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;

    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.lineWidth = 2;

    // pose
    const poseLm = poseRes?.landmarks?.[0];
    if (poseLm?.length) {
      const p = poseLm.slice(0, 25);
      ctx.strokeStyle = "rgba(0, 255, 255, 0.70)";
      drawConnections(ctx, p, POSE_CONNECTIONS, w, h, { mirror: MIRROR });
      ctx.fillStyle = "rgba(0, 255, 255, 0.95)";
      drawPoints(ctx, p, w, h, { mirror: MIRROR, r: 3 });
    }

    // hands
    const list = handRes?.landmarks || [];
    for (const lm of list) {
      ctx.strokeStyle = "rgba(0, 255, 0, 0.70)";
      drawConnections(ctx, lm, HAND_CONNECTIONS, w, h, { mirror: MIRROR });
      ctx.fillStyle = "rgba(0, 255, 0, 0.95)";
      drawPoints(ctx, lm, w, h, { mirror: MIRROR, r: 3 });
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

    // reset
    rawFramesRef.current = [];
    finalFramesRef.current = [];
    setRawCount(0);
    setFramesCount(0);
    setMsLeft(0);

    armedRef.current = false;
    recordingRef.current = false;
    setArmedUI(false);
    setRecordingUI(false);
    setCaptured(false);

    setLabel("-");
    setText("-");
    setConfidence(0);
    setTop5UI([]);

    lastLeftRef.current = [...ZERO63];
    lastRightRef.current = [...ZERO63];
    missLeftRef.current = 0;
    missRightRef.current = 0;

    setCamOn(true);

    const loop = () => {
      rafRef.current = requestAnimationFrame(loop);

      const hand = handRef.current;
      const pose = poseRef.current;
      if (!hand || !pose) return;
      if (video.readyState < 2) return;

      const w = video.videoWidth || 640;
      const h = video.videoHeight || 480;
      const now = performance.now();

      const handRes = hand.detectForVideo(video, now);
      const poseRes = pose.detectForVideo(video, now);

      drawOverlay({ handRes, poseRes, w, h });

      // parse hands (2손)
      let left = null;
      let right = null;

      const handCount = handRes?.landmarks?.length ? handRes.landmarks.length : 0;
      const handDetected = handCount > 0;

      if (handDetected) {
        for (let i = 0; i < handRes.landmarks.length; i++) {
          const lm = handRes.landmarks[i];
          const handed = handRes.handednesses?.[i]?.[0]?.categoryName; // "Left"/"Right"
          const flat = toFlatXYZNormalized(lm);

          if (handed === "Left") left = flat;
          else if (handed === "Right") right = flat;
          else {
            if (!left) left = flat;
            else if (!right) right = flat;
          }
        }
      }

      // hold-last up to 5 frames
      if (left) {
        lastLeftRef.current = left;
        missLeftRef.current = 0;
      } else {
        missLeftRef.current += 1;
      }
      if (right) {
        lastRightRef.current = right;
        missRightRef.current = 0;
      } else {
        missRightRef.current += 1;
      }

      const leftFinal = missLeftRef.current <= 5 ? lastLeftRef.current : ZERO63;
      const rightFinal = missRightRef.current <= 5 ? lastRightRef.current : ZERO63;

      const handActive =
        (leftFinal && leftFinal !== ZERO63) || (rightFinal && rightFinal !== ZERO63);

      // pose
      let poseFlat = ZERO75;
      if (poseRes?.landmarks?.[0]?.length) {
        const p = poseRes.landmarks[0].slice(0, 25);
        poseFlat = toFlatXYZNormalized(p);
      }
      const faceFlat = ZERO210;

      // armed -> start capture
      if (armedRef.current && !recordingRef.current) {
        if (handDetected && handActive) {
          rawFramesRef.current = [];
          finalFramesRef.current = [];
          setRawCount(0);
          setFramesCount(0);

          captureStartRef.current = now;
          captureEndRef.current = now + CAPTURE_MS;
          lastRawPushRef.current = 0;

          recordingRef.current = true;
          armedRef.current = false;

          setRecordingUI(true);
          setArmedUI(false);
          setCaptured(false);

          setLabel("-");
          setText("-");
          setConfidence(0);
          setTop5UI([]);
        }
      }

      // recording: raw capture
      if (recordingRef.current) {
        const endT = captureEndRef.current;
        const leftMs = Math.max(0, endT - now);
        setMsLeft(Math.ceil(leftMs));

        if (RAW_INTERVAL_MS > 0 && now - lastRawPushRef.current < RAW_INTERVAL_MS) {
          // fps cap
        } else {
          lastRawPushRef.current = now;
          rawFramesRef.current.push({
            t: now,
            pose: [...poseFlat],
            face: [...faceFlat],
            leftHand: [...leftFinal],
            rightHand: [...rightFinal],
          });
          setRawCount(rawFramesRef.current.length);
        }

        if (now >= endT) {
          recordingRef.current = false;
          setRecordingUI(false);
          setMsLeft(0);

          const startT = captureStartRef.current;
          const raw = rawFramesRef.current;
          const sampled = resampleByTime(raw, startT, endT, TARGET_FRAMES);

          finalFramesRef.current = sampled;
          setFramesCount(sampled.length);
          setCaptured(true);
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
    setCaptured(false);

    setRawCount(0);
    setFramesCount(0);
    setMsLeft(0);

    if (rafRef.current) cancelAnimationFrame(rafRef.current);

    const video = videoRef.current;
    if (video?.srcObject) {
      const tracks = video.srcObject.getTracks();
      tracks.forEach((t) => t.stop());
      video.srcObject = null;
    }

    // clear canvas
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext?.("2d");
    if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);

    rawFramesRef.current = [];
    finalFramesRef.current = [];
  }

  function record() {
    if (!camOn) return;
    if (armedRef.current || recordingRef.current) return;

    // reset hold-last so old hand doesn't trigger
    lastLeftRef.current = [...ZERO63];
    lastRightRef.current = [...ZERO63];
    missLeftRef.current = 999;
    missRightRef.current = 999;

    rawFramesRef.current = [];
    finalFramesRef.current = [];
    setRawCount(0);
    setFramesCount(0);
    setMsLeft(0);

    armedRef.current = true;
    recordingRef.current = false;
    setArmedUI(true);
    setRecordingUI(false);
    setCaptured(false);
  }

  async function sendPredict() {
    if (!captured) return;
    if (inFlightRef.current) return;

    const frames = deepCloneFrames(finalFramesRef.current);
    if (frames.length !== TARGET_FRAMES) return;

    inFlightRef.current = true;

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frames }),
      });

      if (!res.ok) {
        const t = await res.text();
        throw new Error(`Predict failed ${res.status}: ${t}`);
      }

      const data = await res.json();
      const outLabel = data?.label ?? "-";
      const outText = data?.text ?? outLabel;
      const outConf = Number(data?.confidence ?? 0);

      setLabel(outLabel);
      setText(outText);
      setConfidence(Number.isFinite(outConf) ? outConf : 0);

      const t5 = Array.isArray(data?.top5)
        ? data.top5.map((obj) => {
            const k = Object.keys(obj || {})[0];
            const v = k ? obj[k] : null;
            return { label: k ?? "-", score: typeof v === "number" ? v : null };
          })
        : [];
      setTop5UI(t5);
    } catch (e) {
      console.error(e);
    } finally {
      inFlightRef.current = false;
    }
  }

  const progressPct =
    recordingUI && captureStartRef.current > 0
      ? Math.min(100, Math.round(((CAPTURE_MS - msLeft) / CAPTURE_MS) * 100))
      : captured
      ? 100
      : 0;

  return (
    <div style={{ padding: 16 }}>
      <h2>TranslatorCam (1s / T=60 / Landmark Overlay)</h2>

      <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
        <div>
          <div style={{ position: "relative", width: 560 }}>
            <video
              ref={videoRef}
              playsInline
              muted
              style={{
                width: 560,
                borderRadius: 12,
                background: "#111",
                display: "block",
                transform: MIRROR ? "scaleX(-1)" : "none",
              }}
            />
            <canvas
              ref={canvasRef}
              style={{
                position: "absolute",
                inset: 0,
                width: "100%",
                height: "100%",
                borderRadius: 12,
                pointerEvents: "none",
                transform: MIRROR ? "scaleX(-1)" : "none",
              }}
            />
          </div>

          <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
            <button disabled={!ready || camOn} onClick={startCam}>
              Start Cam
            </button>
            <button disabled={!camOn} onClick={stopCam}>
              Stop Cam
            </button>
            <button disabled={!camOn || armedUI || recordingUI} onClick={record}>
              Record (1s)
            </button>
            <button disabled={!captured || recordingUI || armedUI} onClick={sendPredict}>
              Send (Translate)
            </button>
          </div>

          <div style={{ marginTop: 10, fontSize: 14, opacity: 0.9 }}>
            status: {camOn ? "CAM ON" : "CAM OFF"} | armed: {String(armedUI)} | recording:{" "}
            {String(recordingUI)} | captured: {String(captured)}
            <br />
            rawFrames: {rawCount} | finalFrames: {framesCount}/{TARGET_FRAMES} | msLeft:{" "}
            {recordingUI ? msLeft : 0}
          </div>

          <div style={{ height: 10, background: "#eee", borderRadius: 999, overflow: "hidden", marginTop: 8 }}>
            <div style={{ width: `${progressPct}%`, height: "100%", background: "#111" }} />
          </div>
        </div>

        <div style={{ minWidth: 320 }}>
          <div style={{ fontSize: 14, opacity: 0.8 }}>label</div>
          <div style={{ fontSize: 24, fontWeight: 700 }}>{label}</div>

          <div style={{ marginTop: 12, fontSize: 14, opacity: 0.8 }}>text</div>
          <div style={{ fontSize: 28, fontWeight: 800 }}>{text}</div>

          <div style={{ marginTop: 12, fontSize: 14, opacity: 0.8 }}>confidence</div>
          <div style={{ fontSize: 18, fontWeight: 700 }}>
            {Number.isFinite(confidence) ? confidence.toFixed(6) : "0.000000"}
          </div>

          <div style={{ marginTop: 16, fontSize: 14, opacity: 0.8 }}>top5</div>
          <ol style={{ marginTop: 6 }}>
            {top5UI.map((x, idx) => (
              <li key={`${x.label}-${idx}`}>
                {x.label} {typeof x.score === "number" ? `(${x.score.toFixed(6)})` : ""}
              </li>
            ))}
          </ol>
        </div>
      </div>
    </div>
  );
}
