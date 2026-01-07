"use client";

import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

// ===== 새 학습 스펙 =====
const CAPTURE_MS = 1000;     // 1초
const TARGET_FRAMES = 60;    // T=60 (모델도 60으로 학습)
const RAW_FPS_CAP = 60;      // raw 수집 상한 (0이면 rAF 그대로)
const RAW_INTERVAL_MS = RAW_FPS_CAP > 0 ? 1000 / RAW_FPS_CAP : 0;

// fixed sizes (x,y,z) 3채널
const ZERO75 = Array(25 * 3).fill(0);  // pose 25*3
const ZERO63 = Array(21 * 3).fill(0);  // hand 21*3
const ZERO210 = Array(70 * 3).fill(0); // face placeholder (70*3)

// 셀피처럼 보이게 할지
const MIRROR = false;

// ===== overlay connections =====
const HAND_CONNECTIONS = [
  [0, 1],[1, 2],[2, 3],[3, 4],
  [0, 5],[5, 6],[6, 7],[7, 8],
  [0, 9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
];

const POSE_CONNECTIONS_0_24 = [
  [11, 12],
  [11, 13],[13, 15],
  [12, 14],[14, 16],
  [11, 23],[12, 24],
  [23, 24],
  [23, 25],[25, 27],
  [24, 26],[26, 28],
];

// ===== (x,y,z)로 저장 =====
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

function downloadJson(obj, filename) {
  const blob = new Blob([JSON.stringify(obj)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ===== 시간축 리샘플 =====
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

// ===== overlay draw utils =====
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

// 라벨(예시)
const LABELS = [
  { id: "me", ko: "나(저)" },
  { id: "yes", ko: "네" },
  { id: "no", ko: "아니요" },
  { id: "stop", ko: "멈춰" },
  { id: "go", ko: "가다" },
  { id: "eat", ko: "먹다" },
  { id: "drink", ko: "마시다" },
  { id: "fine", ko: "괜찮아" },
  { id: "wait", ko: "기다려" },
  { id: "call", ko: "전화" },
];

export default function CollectPage() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const rafRef = useRef(null);

  const handRef = useRef(null);
  const poseRef = useRef(null);

  const [ready, setReady] = useState(false);
  const [running, setRunning] = useState(false);

  const [selectedLabel, setSelectedLabel] = useState(LABELS[0].id);
  const [note, setNote] = useState("");

  // 상태: armed(손 대기) -> recording(1초 캡처) -> captured(60프레임 완료)
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

  // hold-last hands
  const lastLeftRef = useRef([...ZERO63]);
  const lastRightRef = useRef([...ZERO63]);
  const missLeftRef = useRef(0);
  const missRightRef = useRef(0);

  // 라벨별 파일 카운터
  function counterKey(label) {
    return `collector_seq_${label}`;
  }
  function nextSeq(label) {
    const key = counterKey(label);
    const cur = Number(localStorage.getItem(key) ?? "0");
    const next = cur + 1;
    localStorage.setItem(key, String(next));
    return next;
  }
  function pad6(n) {
    return String(n).padStart(6, "0");
  }

  useEffect(() => {
    let cancelled = false;

    async function init() {
      const fileset = await FilesetResolver.forVisionTasks(WASM_URL);

      const hand = await HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: HAND_MODEL },
        runningMode: "VIDEO",
        numHands: 1, // 한손 수집이면 1
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

  function clearCanvas() {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext?.("2d");
    if (!canvas || !ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

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
      drawConnections(ctx, p, POSE_CONNECTIONS_0_24, w, h, { mirror: MIRROR });
      ctx.fillStyle = "rgba(0, 255, 255, 0.95)";
      drawPoints(ctx, p, w, h, { mirror: MIRROR, r: 3 });
    }

    // hand
    if (handRes?.landmarks?.length) {
      const lm = handRes.landmarks[0];
      ctx.strokeStyle = "rgba(0, 255, 0, 0.70)";
      drawConnections(ctx, lm, HAND_CONNECTIONS, w, h, { mirror: MIRROR });
      ctx.fillStyle = "rgba(0, 255, 0, 0.95)";
      drawPoints(ctx, lm, w, h, { mirror: MIRROR, r: 3 });
    }

    ctx.restore();
  }

  async function startCam() {
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

    lastLeftRef.current = [...ZERO63];
    lastRightRef.current = [...ZERO63];
    missLeftRef.current = 0;
    missRightRef.current = 0;

    setRunning(true);

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

      // --- hand flatten (한손이면 0번만)
      let left = null;
      let right = null;

      const handDetected = !!handRes?.landmarks?.length;
      if (handDetected) {
        const lm = handRes.landmarks[0];
        const handed = handRes.handednesses?.[0]?.[0]?.categoryName; // "Left"/"Right"
        const flat = toFlatXYZNormalized(lm);

        if (handed === "Left") left = flat;
        else if (handed === "Right") right = flat;
        else right = flat;
      }

      // hold-last up to 5 frames (짧은 미검출 완화)
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

      // --- pose flatten
      let poseFlat = ZERO75;
      if (poseRes?.landmarks?.[0]?.length) {
        const p = poseRes.landmarks[0].slice(0, 25);
        poseFlat = toFlatXYZNormalized(p);
      }
      const faceFlat = ZERO210;

      // 1) armed 상태에서 손이 처음 잡히면 1초 캡처 시작
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
        }
      }

      // 2) recording: raw를 촘촘히 수집
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
    setRunning(false);

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

    clearCanvas();
    rawFramesRef.current = [];
    finalFramesRef.current = [];
  }

  function record1s() {
    if (!running) return;
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

  function saveJson() {
    if (!captured) return;

    const clip = deepCloneFrames(finalFramesRef.current);
    if (clip.length !== TARGET_FRAMES) return;

    const seq = nextSeq(selectedLabel);
    const filename = `${selectedLabel}_${pad6(seq)}.json`;

    const payload = {
      version: 2,
      createdAt: new Date().toISOString(),
      label: selectedLabel,
      labelKo: LABELS.find((x) => x.id === selectedLabel)?.ko ?? selectedLabel,
      note: note.trim() || "",
      frames: clip,
      meta: {
        capture_ms: CAPTURE_MS,
        target_frames: TARGET_FRAMES,
        raw_fps_cap: RAW_FPS_CAP,
        mirror: MIRROR,
        models: { hand: HAND_MODEL, pose: POSE_MODEL },
        channel: "xyz",
      },
    };

    downloadJson(payload, filename);

    // next
    rawFramesRef.current = [];
    finalFramesRef.current = [];
    setRawCount(0);
    setFramesCount(0);
    setMsLeft(0);
    setCaptured(false);
  }

  const progressPct =
    recordingUI && captureStartRef.current > 0
      ? Math.min(100, Math.round(((CAPTURE_MS - msLeft) / CAPTURE_MS) * 100))
      : captured
      ? 100
      : 0;

  return (
    <div style={{ padding: 16 }}>
      <h2>Dataset Collector (1s / T=60 / Landmark Overlay)</h2>

      <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
        <div>
          <div style={{ marginBottom: 10, display: "flex", gap: 8, alignItems: "center" }}>
            <select
              value={selectedLabel}
              onChange={(e) => setSelectedLabel(e.target.value)}
              disabled={running && (armedUI || recordingUI)}
              style={{ width: 220, padding: "8px 10px", borderRadius: 8, border: "1px solid #444" }}
            >
              {LABELS.map((x) => (
                <option key={x.id} value={x.id}>
                  {x.ko} ({x.id})
                </option>
              ))}
            </select>

            <input
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="옵션 메모 (예: fast / dark / left-hand)"
              disabled={running && (armedUI || recordingUI)}
              style={{ width: 360, padding: "8px 10px", borderRadius: 8, border: "1px solid #444" }}
            />
          </div>

          <div style={{ position: "relative", width: 480 }}>
            <video
              ref={videoRef}
              playsInline
              muted
              style={{
                width: 480,
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
            <button disabled={!ready || running} onClick={startCam}>
              Start Cam
            </button>
            <button disabled={!running} onClick={stopCam}>
              Stop Cam
            </button>

            <button disabled={!running || armedUI || recordingUI} onClick={record1s}>
              Record (1s)
            </button>

            <button disabled={!captured || recordingUI || armedUI} onClick={saveJson}>
              Save JSON
            </button>
          </div>

          <div style={{ marginTop: 10, fontSize: 14, opacity: 0.9 }}>
            status: {running ? "CAM ON" : "CAM OFF"} | armed: {String(armedUI)} | recording:{" "}
            {String(recordingUI)} | captured: {String(captured)}
            <br />
            rawFrames: {rawCount} | finalFrames: {framesCount}/{TARGET_FRAMES} | msLeft:{" "}
            {recordingUI ? msLeft : 0}
          </div>

          <div style={{ height: 10, background: "#eee", borderRadius: 999, overflow: "hidden", marginTop: 8 }}>
            <div style={{ width: `${progressPct}%`, height: "100%", background: "#111" }} />
          </div>
        </div>

        <div style={{ minWidth: 260 }}>
          <div style={{ fontSize: 14, opacity: 0.8 }}>현재 라벨</div>
          <div style={{ fontSize: 24, fontWeight: 800 }}>
            {LABELS.find((x) => x.id === selectedLabel)?.ko} ({selectedLabel})
          </div>

          <div style={{ marginTop: 12, fontSize: 12, opacity: 0.75, lineHeight: 1.5 }}>
            규칙:
            <br />- Record 누르면 손 감지 대기(armed)
            <br />- 손이 잡히는 순간부터 1초 캡처
            <br />- raw를 시간축 리샘플해서 항상 {TARGET_FRAMES}프레임 저장
            <br />- 좌표는 (x,y,z) normalized로 저장
          </div>
        </div>
      </div>
    </div>
  );
}
