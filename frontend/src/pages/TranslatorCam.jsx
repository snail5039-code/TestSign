import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const FRAMES = 30;

// ✅ 기본은 Spring(8080) -> 내부에서 FastAPI(/predict) 호출하게 해둔 상태
// 필요하면 여기만 바꿔.
const API_URL = "http://127.0.0.1:8080/api/translate";
// const API_URL = "http://127.0.0.1:8000/predict";

// 너의 feature D=411 = pose(25*3=75) + face(70*3=210) + left(63) + right(63)
const ZERO75 = Array(25 * 3).fill(0);
const ZERO63 = Array(21 * 3).fill(0);
const ZERO210 = Array(70 * 3).fill(0);

// ---- utils ----
function clamp01(v) {
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

/**
 * IMPORTANT
 * 학습 데이터(JSON) 값들이 0~1 normalized 형태면,
 * 프론트도 "픽셀로 곱하지 말고" normalized 그대로 보내는 게 맞음.
 *
 * dataset 예시 값이 0.3~0.9대인 걸로 봐서 normalized가 맞아 보임.
 */
function toFlatXYC_Normalized(landmarks, confFallback = 1) {
  const out = [];
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i] || {};
    const x = clamp01(lm.x ?? 0);
    const y = clamp01(lm.y ?? 0);
    // 너 포맷(3번째 값) = visibility/presence/1
    const c =
      (Number.isFinite(lm.visibility) ? lm.visibility : undefined) ??
      (Number.isFinite(lm.presence) ? lm.presence : undefined) ??
      confFallback ??
      1;
    out.push(x, y, c);
  }
  return out;
}

function energy(arr) {
  if (!Array.isArray(arr) || arr.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += Math.abs(arr[i] || 0);
  return s;
}

/**
 * hold-last:
 * - 랜드마크가 1~2프레임 끊길 때 0으로 떨어지는 걸 방지
 */
function chooseWithHold(current, last, holdLeftRef, holdMax = 6) {
  const e = energy(current);
  if (e > 0) {
    holdLeftRef.current = holdMax;
    return current;
  }
  if (holdLeftRef.current > 0 && energy(last) > 0) {
    holdLeftRef.current -= 1;
    return last;
  }
  return current;
}

export default function TranslatorCam() {
  const videoRef = useRef(null);
  const rafRef = useRef(null);

  const handRef = useRef(null);
  const poseRef = useRef(null);

  const [ready, setReady] = useState(false);
  const [running, setRunning] = useState(false);

  const framesRef = useRef([]); // 캡처된 프레임 버퍼
  const sentFramesRef = useRef(0); // Stop 때 보낸 프레임 수

  // 결과 UI
  const [label, setLabel] = useState("-");
  const [text, setText] = useState("-");
  const [confidence, setConfidence] = useState(0);
  const [top5UI, setTop5UI] = useState([]);

  // 디버그 UI
  const [dbg, setDbg] = useState({
    poseE: 0,
    leftE: 0,
    rightE: 0,
    handCount: 0,
  });

  // timestamp monotonic (MediaPipe VIDEO 모드에서 중요)
  const lastTsRef = useRef(0);

  // hold-last refs
  const lastPoseRef = useRef(ZERO75);
  const lastLeftRef = useRef(ZERO63);
  const lastRightRef = useRef(ZERO63);
  const holdPoseRef = useRef(0);
  const holdLeftHandRef = useRef(0);
  const holdRightHandRef = useRef(0);

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

  async function startCam() {
    const video = videoRef.current;
    if (!video) return;

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    // reset buffers/state
    framesRef.current = [];
    sentFramesRef.current = 0;
    setLabel("-");
    setText("-");
    setConfidence(0);
    setTop5UI([]);

    lastPoseRef.current = ZERO75;
    lastLeftRef.current = ZERO63;
    lastRightRef.current = ZERO63;
    holdPoseRef.current = 0;
    holdLeftHandRef.current = 0;
    holdRightHandRef.current = 0;

    setRunning(true);

    const loop = () => {
      rafRef.current = requestAnimationFrame(loop);

      const hand = handRef.current;
      const pose = poseRef.current;
      if (!hand || !pose) return;
      if (video.readyState < 2) return;

      // monotonic timestamp 보장
      let ts = performance.now();
      if (ts <= lastTsRef.current) ts = lastTsRef.current + 1;
      lastTsRef.current = ts;

      // ---- HANDS ----
      const handRes = hand.detectForVideo(video, ts);
      let left = ZERO63;
      let right = ZERO63;

      const handCount = Array.isArray(handRes?.landmarks) ? handRes.landmarks.length : 0;

      if (handCount) {
        for (let i = 0; i < handRes.landmarks.length; i++) {
          const lm = handRes.landmarks[i]; // 21 pts
          const handed = handRes.handednesses?.[i]?.[0]?.categoryName; // "Left"/"Right"
          const flat = toFlatXYC_Normalized(lm, 1);

          if (handed === "Left") left = flat;
          else if (handed === "Right") right = flat;
          else {
            // fallback
            if (energy(left) === 0) left = flat;
            else if (energy(right) === 0) right = flat;
          }
        }
      }

      // hold-last 적용 (한두 프레임 끊겨도 0으로 갑자기 떨어지는 것 방지)
      left = chooseWithHold(left, lastLeftRef.current, holdLeftHandRef, 6);
      right = chooseWithHold(right, lastRightRef.current, holdRightHandRef, 6);
      lastLeftRef.current = left;
      lastRightRef.current = right;

      // ---- POSE ----
      const poseRes = pose.detectForVideo(video, ts);
      let poseFlat = ZERO75;

      if (poseRes?.landmarks?.[0]?.length) {
        const p = poseRes.landmarks[0].slice(0, 25);
        poseFlat = toFlatXYC_Normalized(p, 1);
      }

      poseFlat = chooseWithHold(poseFlat, lastPoseRef.current, holdPoseRef, 6);
      lastPoseRef.current = poseFlat;

      // ---- FACE (일단 0 채움) ----
      const faceFlat = ZERO210;

      const frame = {
        pose: poseFlat,
        face: faceFlat,
        leftHand: left,
        rightHand: right,
      };

      const buf = framesRef.current;
      buf.push(frame);
      if (buf.length > FRAMES) buf.shift();

      // debug
      setDbg({
        poseE: energy(poseFlat),
        leftE: energy(left),
        rightE: energy(right),
        handCount,
      });
    };

    rafRef.current = requestAnimationFrame(loop);
  }

  async function stopAndPredict() {
    // 1) 루프/캠 정지
    setRunning(false);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);

    const video = videoRef.current;
    if (video?.srcObject) {
      const tracks = video.srcObject.getTracks();
      tracks.forEach((t) => t.stop());
      video.srcObject = null;
    }

    // 2) 마지막 FRAMES 프레임 준비(부족하면 0패딩)
    const buf = framesRef.current || [];
    const frames = buf.slice(-FRAMES);

    while (frames.length < FRAMES) {
      frames.unshift({
        pose: ZERO75,
        face: ZERO210,
        leftHand: ZERO63,
        rightHand: ZERO63,
      });
    }

    sentFramesRef.current = frames.length;

    // 디버그 로그 (네가 보던 것처럼 frames 같이 확인 가능)
    const last = frames[frames.length - 1];
    console.log("[STOP] frames len =", frames.length);
    console.log("[STOP] lens pose/face/left/right =", last.pose.length, last.face.length, last.leftHand.length, last.rightHand.length);
    console.log("[STOP] energy pose/left/right =", energy(last.pose), energy(last.leftHand), energy(last.rightHand));
    console.log("[STOP] sample first6 left/pose/right =", last.leftHand.slice(0, 6), last.pose.slice(0, 6), last.rightHand.slice(0, 6));

    // 3) 서버로 1번만 호출
    await sendPredict(frames);
  }

  async function sendPredict(frames) {
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
      console.log("[PREDICT] response:", data);

      // 서버 응답 형태(네 스샷 기준):
      // {
      //   "label": "call",
      //   "confidence": 0.998...,
      //   "text": "call",
      //   "top5": [ {"call":0.99}, {"drink":0.001}, ... ],
      //   "pred_idx": 0
      // }

      const outLabel = data?.label ?? data?.pred ?? "-";
      const outText = data?.text ?? outLabel ?? "-";
      const outConf = Number(data?.confidence ?? data?.score ?? 0);

      setLabel(String(outLabel));
      setText(String(outText));
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
      console.error("Error during predict:", e);
    }
  }

  const curCount = framesRef.current.length;
  const progressPct = Math.round((Math.min(curCount, FRAMES) / FRAMES) * 100);

  return (
    <div style={{ padding: 16 }}>
      <h2 style={{ margin: 0, marginBottom: 12 }}>TranslatorCam</h2>

      <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
        {/* LEFT: video + controls */}
        <div style={{ width: 520 }}>
          <div
            style={{
              width: 480,
              height: 270,
              borderRadius: 16,
              background: "#111",
              overflow: "hidden",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <video
              ref={videoRef}
              playsInline
              muted
              style={{ width: 480, height: 270, objectFit: "cover" }}
            />
          </div>

          <div style={{ marginTop: 12, display: "flex", gap: 8, alignItems: "center" }}>
            <button disabled={!ready || running} onClick={startCam}>
              Start
            </button>
            <button disabled={!ready || !running} onClick={stopAndPredict}>
              Stop (Predict)
            </button>
            <div style={{ fontSize: 12, opacity: 0.75, marginLeft: 8 }}>
              ready: {String(ready)} / running: {String(running)}
            </div>
          </div>

          {/* frames progress */}
          <div style={{ marginTop: 10 }}>
            <div style={{ fontSize: 12, opacity: 0.8, marginBottom: 6 }}>
              frames: {Math.min(curCount, FRAMES)}/{FRAMES} {running ? "(capturing)" : "(stopped)"}{" "}
              | sentFrames(last stop): {sentFramesRef.current}
            </div>
            <div
              style={{
                width: 480,
                height: 10,
                borderRadius: 999,
                background: "#eee",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${progressPct}%`,
                  height: "100%",
                  background: "#111",
                }}
              />
            </div>
          </div>

          {/* debug */}
          <div style={{ marginTop: 12, fontSize: 12, opacity: 0.8 }}>
            debug: poseE={dbg.poseE.toFixed(2)} / leftE={dbg.leftE.toFixed(2)} / rightE={dbg.rightE.toFixed(2)}{" "}
            / handCount={dbg.handCount}
            <div style={{ marginTop: 6, opacity: 0.7 }}>
              (랜드마크가 잠깐 0 되는 문제는 hold-last로 완화됨)
            </div>
          </div>
        </div>

        {/* RIGHT: results */}
        <div style={{ minWidth: 360 }}>
          <div style={{ fontSize: 13, opacity: 0.8 }}>label</div>
          <div style={{ fontSize: 28, fontWeight: 800, marginBottom: 10 }}>{label}</div>

          <div style={{ fontSize: 13, opacity: 0.8 }}>text</div>
          <div style={{ fontSize: 28, fontWeight: 800, marginBottom: 10 }}>{text}</div>

          <div style={{ fontSize: 13, opacity: 0.8 }}>confidence</div>
          <div style={{ fontSize: 20, fontWeight: 700, marginBottom: 14 }}>
            {Number.isFinite(confidence) ? confidence.toFixed(6) : "0.000000"}
          </div>

          <div style={{ fontSize: 13, opacity: 0.8 }}>top5</div>
          <ol style={{ marginTop: 8 }}>
            {top5UI.map((x, idx) => (
              <li key={`${x.label}-${idx}`}>
                {x.label}{" "}
                {typeof x.score === "number" ? `(${x.score.toFixed(6)})` : ""}
              </li>
            ))}
          </ol>

          <div style={{ marginTop: 10, fontSize: 12, opacity: 0.7, lineHeight: 1.5 }}>
            ※ Stop을 누를 때 마지막 {FRAMES}프레임을 서버로 보내서 예측합니다.
            <br />
            ※ face는 현재 0으로 채움(동작 우선). 다음 단계에서 face 70개 매핑 붙이면 정확도 더 올라감.
          </div>
        </div>
      </div>
    </div>
  );
}
