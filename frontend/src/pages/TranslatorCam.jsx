import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const FRAMES = 30; // 30프레임 녹화

// fixed sizes
const ZERO75 = Array(25 * 3).fill(0); // pose 25*3
const ZERO63 = Array(21 * 3).fill(0); // hand 21*3
const ZERO210 = Array(70 * 3).fill(0); // face 70*3 (일단 0, 다음 단계에서 매핑 붙임)

// ✅ 셀피 카메라처럼 좌우 반전해서 보고 싶으면 true
const MIRROR = false;

// ✅ 손 연결(21개 랜드마크 연결선)
const HAND_CONNECTIONS = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [0, 17],
];

// ✅ 포즈(0~24만 쓰는 현재 코드에 맞춘 상체 위주 연결선)
const POSE_CONNECTIONS_0_24 = [
  [11, 12], // shoulders
  [11, 13],
  [13, 15], // left arm
  [12, 14],
  [14, 16], // right arm
  [11, 23],
  [12, 24], // shoulders -> hips
  [23, 24], // hips
  [15, 17],
  [15, 19],
  [15, 21], // left wrist -> fingers
  [16, 18],
  [16, 20],
  [16, 22], // right wrist -> fingers
];

function toFlatXYZ(landmarks, w, h, conf = 1) {
  const out = [];
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    const x = (lm?.x ?? 0) * w;
    const y = (lm?.y ?? 0) * h;
    const c = (lm?.visibility ?? lm?.presence ?? conf ?? 1) || 1;
    out.push(x, y, c);
  }
  return out;
}

// ✅ 그리기 유틸
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

  const [ready, setReady] = useState(false);
  const [running, setRunning] = useState(false);

  const [text, setText] = useState("-");
  const [label, setLabel] = useState("-");
  const [conf, setConf] = useState(0);

  const framesRef = useRef([]); // last 30 frames
  const inFlightRef = useRef(false); // avoid spam
  const lastSendAtRef = useRef(0);

  // ✅ 저장 버튼 활성화/표시용
  const [framesCount, setFramesCount] = useState(0);

  // ✅ 키워드 입력 + 촬영 시작 시 고정될 키워드
  const [keyword, setKeyword] = useState("");
  const keywordFixedRef = useRef(""); // Start 시점의 keyword를 고정 저장

  // ✅ 파일명 숫자 시퀀스 (localStorage에 유지)
  const COUNTER_KEY = "translatorcam_clip_seq";
  function nextSeq() {
    const cur = Number(localStorage.getItem(COUNTER_KEY) ?? "0");
    const next = cur + 1;
    localStorage.setItem(COUNTER_KEY, String(next));
    return next;
  }
  function pad6(n) {
    return String(n).padStart(6, "0");
  }

  // streak smoothing
  const lastLabelRef = useRef(null);
  const streakRef = useRef(0);

  const handRef = useRef(null);
  const poseRef = useRef(null);

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

  // ✅ JSON 다운로드 유틸 + 저장 함수 (Save 버튼 눌렀을 때만 실행)
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

  function saveClipToFile() {
    const clip = framesRef.current.slice(0, FRAMES);
    if (clip.length < FRAMES) return;

    const seq = nextSeq(); // ✅ 1,2,3...
    const filename = `${pad6(seq)}.json`; // ✅ 000001.json, 000002.json ...

    const now = new Date();
    const payload = {
      version: 1,
      seq, // ✅ 파일명과 매칭
      createdAt: now.toISOString(),
      keyword: keywordFixedRef.current, // ✅ 입력 키워드 같이 저장
      frames: clip,
      lastPrediction: { label, text, confidence: conf },
    };

    downloadJson(payload, filename);
  }

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

    // 캔버스 픽셀 좌표계를 비디오 원본 해상도에 맞춤
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;

    ctx.clearRect(0, 0, w, h);

    ctx.save();
    ctx.lineWidth = 2;

    // --- pose
    const poseLm = poseRes?.landmarks?.[0];
    if (poseLm?.length) {
      const p = poseLm.slice(0, 25);

      ctx.strokeStyle = "rgba(0, 255, 255, 0.75)";
      drawConnections(ctx, p, POSE_CONNECTIONS_0_24, w, h, { mirror: MIRROR });

      ctx.fillStyle = "rgba(0, 255, 255, 0.95)";
      drawPoints(ctx, p, w, h, { mirror: MIRROR, r: 3 });
    }

    // --- hands
    if (handRes?.landmarks?.length) {
      for (let i = 0; i < handRes.landmarks.length; i++) {
        const lm = handRes.landmarks[i];
        const handed = handRes.handednesses?.[i]?.[0]?.categoryName; // "Left"/"Right"

        const color =
          handed === "Left"
            ? { stroke: "rgba(255, 0, 255, 0.75)", fill: "rgba(255, 0, 255, 0.95)" }
            : handed === "Right"
            ? { stroke: "rgba(0, 255, 0, 0.75)", fill: "rgba(0, 255, 0, 0.95)" }
            : { stroke: "rgba(255, 255, 0, 0.75)", fill: "rgba(255, 255, 0, 0.95)" };

        ctx.strokeStyle = color.stroke;
        drawConnections(ctx, lm, HAND_CONNECTIONS, w, h, { mirror: MIRROR });

        ctx.fillStyle = color.fill;
        drawPoints(ctx, lm, w, h, { mirror: MIRROR, r: 3 });
      }
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

      // --- detect hands
      const handRes = hand.detectForVideo(video, now);
      let left = ZERO63;
      let right = ZERO63;

      if (handRes?.landmarks?.length) {
        for (let i = 0; i < handRes.landmarks.length; i++) {
          const lm = handRes.landmarks[i];
          const handed = handRes.handednesses?.[i]?.[0]?.categoryName; // "Left"/"Right"
          const flat = toFlatXYZ(lm, w, h, 1);
          if (handed === "Left") left = flat;
          else if (handed === "Right") right = flat;
          else {
            if (left === ZERO63) left = flat;
            else if (right === ZERO63) right = flat;
          }
        }
      }

      // --- detect pose
      const poseRes = pose.detectForVideo(video, now);
      let poseFlat = ZERO75;
      if (poseRes?.landmarks?.[0]?.length) {
        const p = poseRes.landmarks[0].slice(0, 25);
        poseFlat = toFlatXYZ(p, w, h, 1);
      }

      // ✅ 여기서 오버레이 그리기
      drawOverlay({ handRes, poseRes, w, h });

      // --- face (0으로 임시 채움)
      const faceFlat = ZERO210;

      // --- combine frame
      const frame = { pose: poseFlat, face: faceFlat, leftHand: left, rightHand: right };

      // ** 손, 포즈가 모두 있는 경우에만 프레임 추가 **
      if (left !== ZERO63 && right !== ZERO63 && poseFlat !== ZERO75) {
        const buf = framesRef.current;
        buf.push(frame);
        if (buf.length > FRAMES) buf.shift();
        setFramesCount(buf.length);
      }

      // 30프레임 차면 번역 요청
      const canSend =
        framesRef.current.length === FRAMES && !inFlightRef.current && now - lastSendAtRef.current > 200;
      if (canSend) {
        lastSendAtRef.current = now;
        sendTranslate([...framesRef.current]);
      }
    };

    rafRef.current = requestAnimationFrame(loop);
  }

  function stopCam() {
    setRunning(false);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);

    const video = videoRef.current;
    if (video?.srcObject) {
      const tracks = video.srcObject.getTracks();
      tracks.forEach((t) => t.stop());
      video.srcObject = null;
    }

    clearCanvas();
    // ✅ Stop은 저장 절대 안 함
  }

  async function sendTranslate(frames) {
    inFlightRef.current = true;
    try {
      const res = await fetch("/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frames }),
      });

      const data = await res.json();
      console.log("Server response:", data);

      const curLabel = data?.label ?? null;
      if (curLabel && curLabel === lastLabelRef.current) streakRef.current += 1;
      else {
        lastLabelRef.current = curLabel;
        streakRef.current = 1;
      }

      if ((data?.confidence ?? 0) >= 0.85 && streakRef.current >= 3) {
        setLabel(data.label);
        setText(data.text ?? data.label);
        setConf(data.confidence ?? 0);
      } else {
        setConf(data?.confidence ?? 0);
      }
    } catch (e) {
      console.error("Error during translation:", e);
    } finally {
      inFlightRef.current = false;
    }
  }

  return (
    <div style={{ padding: 16 }}>
      <h2>TranslatorCam</h2>

      <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
        <div>
          {/* ✅ 키워드 입력 */}
          <div style={{ marginBottom: 10, display: "flex", gap: 8, alignItems: "center" }}>
            <input
              value={keyword}
              onChange={(e) => setKeyword(e.target.value)}
              placeholder="키워드 입력 후 Start"
              style={{ width: 480, padding: "8px 10px", borderRadius: 8, border: "1px solid #444" }}
              disabled={running}
            />
          </div>

          {/* ✅ video + canvas overlay */}
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
            <button disabled={!ready || running || !keyword.trim()} onClick={startCam}>
              Start
            </button>
            <button disabled={!running} onClick={stopCam}>
              Stop
            </button>

            {/* ✅ 30프레임 꽉 찼을 때만 저장 가능 */}
            <button disabled={framesCount < FRAMES} onClick={saveClipToFile}>
              Save JSON
            </button>
          </div>

          <div style={{ marginTop: 10, fontSize: 14, opacity: 0.9 }}>
            frames: {framesCount}/{FRAMES} | conf: {conf.toFixed(3)}
          </div>
        </div>

        <div style={{ minWidth: 260 }}>
          <div style={{ fontSize: 14, opacity: 0.8 }}>label</div>
          <div style={{ fontSize: 24, fontWeight: 700 }}>{label}</div>

          <div style={{ marginTop: 12, fontSize: 14, opacity: 0.8 }}>text</div>
          <div style={{ fontSize: 28, fontWeight: 800 }}>{text}</div>

          <div style={{ marginTop: 12, fontSize: 12, opacity: 0.7 }}>
            ※ 현재 face는 0으로 채움(동작 우선). 다음 단계에서 AIHub 70 face 매핑 붙이면 정확도 더 올라감.
          </div>
        </div>
      </div>
    </div>
  );
}
