"use client";

import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const FRAMES = 30;

// fixed sizes
const ZERO75 = Array(25 * 3).fill(0); // pose 25*3 (x,y,conf)
const ZERO63 = Array(21 * 3).fill(0); // hand 21*3
const ZERO210 = Array(70 * 3).fill(0); // face placeholder (70*3)

// 수집은 한손이 목적이면 MIRROR는 취향 (대개 셀피처럼 보려면 true)
const MIRROR = false;

// connections (overlay)
const HAND_CONNECTIONS = [
  [0, 1],[1, 2],[2, 3],[3, 4],
  [0, 5],[5, 6],[6, 7],[7, 8],
  [5, 9],[9, 10],[10, 11],[11, 12],
  [9, 13],[13, 14],[14, 15],[15, 16],
  [13, 17],[17, 18],[18, 19],[19, 20],
  [0, 17],
];

const POSE_CONNECTIONS_0_24 = [
  [11, 12],
  [11, 13],[13, 15],
  [12, 14],[14, 16],
  [11, 23],[12, 24],
  [23, 24],
  [15, 17],[15, 19],[15, 21],
  [16, 18],[16, 20],[16, 22],
];

// ====== 좌표 저장: 정규화(0~1)로 저장 ======
function toFlatXYZNormalized(landmarks, conf = 1) {
  const out = [];
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    const x = lm?.x ?? 0;
    const y = lm?.y ?? 0;
    const c = (lm?.visibility ?? lm?.presence ?? conf ?? 1) || 1;
    out.push(x, y, c);
  }
  return out;
}

// ====== overlay drawing utils ======
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

// 라벨(한손 테스트용 10개)
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
  const [note, setNote] = useState(""); // 옵션 메모 (예: speed=fast 등)

  const framesRef = useRef([]);
  const [framesCount, setFramesCount] = useState(0);

  const isRecordingRef = useRef(false);
  const [isRecording, setIsRecording] = useState(false);

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
        numHands: 1, // ✅ 한손 수집이면 1로 고정 추천
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

    framesRef.current = [];
    setFramesCount(0);

    isRecordingRef.current = false;
    setIsRecording(false);

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

      // --- hand flatten (한손이니까 0번만 사용)
      let left = ZERO63;
      let right = ZERO63;

      if (handRes?.landmarks?.length) {
        const lm = handRes.landmarks[0];
        const handed = handRes.handednesses?.[0]?.[0]?.categoryName; // "Left"/"Right"
        const flat = toFlatXYZNormalized(lm, 1);

        // 저장 포맷은 기존과 호환되게 left/right 둘 다 유지
        if (handed === "Left") left = flat;
        else if (handed === "Right") right = flat;
        else right = flat; // 애매하면 right에 넣어둠
      }

      // --- pose flatten (없으면 ZERO)
      let poseFlat = ZERO75;
      if (poseRes?.landmarks?.[0]?.length) {
        const p = poseRes.landmarks[0].slice(0, 25);
        poseFlat = toFlatXYZNormalized(p, 1);
      }

      const faceFlat = ZERO210;

      // 기록 프레임
      if (isRecordingRef.current) {
        const hasHand = left !== ZERO63 || right !== ZERO63; // ✅ 한손만 있어도 OK

        if (hasHand) {
          const frame = { pose: poseFlat, face: faceFlat, leftHand: left, rightHand: right };
          const buf = framesRef.current;
          buf.push(frame);

          // 목표 프레임 채우면 자동 정지
          if (buf.length >= FRAMES) {
            isRecordingRef.current = false;
            setIsRecording(false);
          }

          // UI 업데이트
          setFramesCount(Math.min(buf.length, FRAMES));
        }
      }
    };

    rafRef.current = requestAnimationFrame(loop);
  }

  function stopCam() {
    setRunning(false);
    isRecordingRef.current = false;
    setIsRecording(false);

    if (rafRef.current) cancelAnimationFrame(rafRef.current);

    const video = videoRef.current;
    if (video?.srcObject) {
      const tracks = video.srcObject.getTracks();
      tracks.forEach((t) => t.stop());
      video.srcObject = null;
    }

    clearCanvas();
    framesRef.current = [];
    setFramesCount(0);
  }

  function startRecord() {
    if (!running) return;
    framesRef.current = [];
    setFramesCount(0);
    isRecordingRef.current = true;
    setIsRecording(true);
  }

  function saveClip() {
    const clip = framesRef.current.slice(0, FRAMES);
    if (clip.length < FRAMES) return;

    const seq = nextSeq(selectedLabel);
    const filename = `${selectedLabel}_${pad6(seq)}.json`;

    const now = new Date();
    const payload = {
      version: 1,
      createdAt: now.toISOString(),
      label: selectedLabel,
      labelKo: LABELS.find((x) => x.id === selectedLabel)?.ko ?? selectedLabel,
      note: note.trim() || "",
      frames: clip,
      meta: {
        frames: FRAMES,
        mirror: MIRROR,
        models: { hand: HAND_MODEL, pose: POSE_MODEL },
      },
    };

    downloadJson(payload, filename);

    // 다음 샘플 준비
    framesRef.current = [];
    setFramesCount(0);
  }

  return (
    <div style={{ padding: 16 }}>
      <h2>Dataset Collector (One-hand)</h2>

      <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
        <div>
          <div style={{ marginBottom: 10, display: "flex", gap: 8, alignItems: "center" }}>
            <select
              value={selectedLabel}
              onChange={(e) => setSelectedLabel(e.target.value)}
              disabled={running && isRecording}
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
              disabled={running && isRecording}
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

            <button disabled={!running || isRecording} onClick={startRecord}>
              Record {FRAMES}f
            </button>

            <button disabled={framesCount < FRAMES || isRecording} onClick={saveClip}>
              Save JSON
            </button>
          </div>

          <div style={{ marginTop: 10, fontSize: 14, opacity: 0.9 }}>
            status: {running ? "CAM ON" : "CAM OFF"} | recording: {isRecording ? "YES" : "NO"} | frames:{" "}
            {framesCount}/{FRAMES}
          </div>
        </div>

        <div style={{ minWidth: 260 }}>
          <div style={{ fontSize: 14, opacity: 0.8 }}>현재 라벨</div>
          <div style={{ fontSize: 24, fontWeight: 800 }}>
            {LABELS.find((x) => x.id === selectedLabel)?.ko} ({selectedLabel})
          </div>

          <div style={{ marginTop: 12, fontSize: 12, opacity: 0.75, lineHeight: 1.5 }}>
            규칙:
            <br />- Record 누르면 버퍼 초기화 후 {FRAMES}프레임만 수집
            <br />- 한 손만 잡혀도 프레임 적재됨
            <br />- 좌표는 0~1 정규화로 저장
          </div>
        </div>
      </div>
    </div>
  );
}
