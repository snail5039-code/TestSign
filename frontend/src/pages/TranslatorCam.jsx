import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const FRAMES = 30;

// fixed sizes
const ZERO75 = Array(25 * 3).fill(0);   // pose 25*3
const ZERO63 = Array(21 * 3).fill(0);   // hand 21*3
const ZERO210 = Array(70 * 3).fill(0);  // face 70*3 (일단 0, 다음 단계에서 매핑 붙임)

function toFlatXYZ(landmarks, w, h, conf = 1) {
  // landmarks: normalized x,y in [0,1]
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

export default function TranslatorCam() {
  const videoRef = useRef(null);
  const rafRef = useRef(null);

  const [ready, setReady] = useState(false);
  const [running, setRunning] = useState(false);

  const [text, setText] = useState("-");
  const [label, setLabel] = useState("-");
  const [conf, setConf] = useState(0);

  const framesRef = useRef([]);      // last 30 frames
  const inFlightRef = useRef(false); // avoid spam
  const lastSendAtRef = useRef(0);

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

      // --- face (0으로 임시 채움, 나중에 AIHub 70 face 매핑 붙임)
      const faceFlat = ZERO210;

      // --- combine frame
      const frame = { pose: poseFlat, face: faceFlat, leftHand: left, rightHand: right };

      const buf = framesRef.current;
      buf.push(frame);
      if (buf.length > FRAMES) buf.shift();

      // 30프레임 차면 번역 요청
      const canSend = buf.length === FRAMES && !inFlightRef.current && now - lastSendAtRef.current > 200;
      if (canSend) {
        lastSendAtRef.current = now;
        sendTranslate([...buf]);
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
  }

  async function sendTranslate(frames) {
    inFlightRef.current = true;
    try {
      const res = await fetch("/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ frames }),  // 모든 데이터를 한 번에 넘김
      });

      const data = await res.json();
      console.log('Server response:', data);  // 서버 응답 확인을 위한 로그

      // streak smoothing: 같은 label 3번 연속이면 확정 표시
      const curLabel = data?.label ?? null;
      if (curLabel && curLabel === lastLabelRef.current) streakRef.current += 1;
      else {
        lastLabelRef.current = curLabel;
        streakRef.current = 1;
      }

      if ((data?.confidence ?? 0) >= 0.85 && streakRef.current >= 3) {
        setLabel(data.label);   // 라벨 업데이트
        setText(data.text ?? data.label);   // 텍스트 업데이트
        setConf(data.confidence ?? 0);   // confidence 업데이트
      } else {
        setConf(data?.confidence ?? 0);
      }
    } catch (e) {
      console.error('Error during translation:', e);  // 에러 로그 추가
    } finally {
      inFlightRef.current = false;
    }
  }

  return (
    <div style={{ padding: 16 }}>
      <h2>TranslatorCam</h2>

      <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
        <div>
          <video
            ref={videoRef}
            playsInline
            muted
            style={{ width: 480, borderRadius: 12, background: "#111" }}
          />
          <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
            <button disabled={!ready || running} onClick={startCam}>
              Start
            </button>
            <button disabled={!running} onClick={stopCam}>
              Stop
            </button>
          </div>
          <div style={{ marginTop: 10, fontSize: 14, opacity: 0.9 }}>
            frames: {framesRef.current.length}/{FRAMES} | conf: {conf.toFixed(3)}
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
