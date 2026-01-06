import { useEffect, useRef, useState } from "react";
import axios from "axios";
import { FilesetResolver, HandLandmarker, PoseLandmarker, FaceLandmarker } from "@mediapipe/tasks-vision";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const FACE_MODEL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const WINDOW = 30; // ✅ AIHub frames=30

export default function TranslatorCam() {
  const videoRef = useRef(null);
  const rafRef = useRef(null);
  const framesRef = useRef([]);
  const lastSendRef = useRef(0);

  const [ready, setReady] = useState(false);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [err, setErr] = useState("");

  const landmarkersRef = useRef({ hand: null, pose: null, face: null });

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_URL);
        const hand = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: HAND_MODEL },
          runningMode: "VIDEO",
          numHands: 2
        });
        const pose = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: POSE_MODEL },
          runningMode: "VIDEO"
        });
        const face = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: FACE_MODEL },
          runningMode: "VIDEO"
        });

        if (!alive) return;
        landmarkersRef.current = { hand, pose, face };
        setReady(true);
      } catch (e) {
        setErr(String(e));
      }
    })();
    return () => { alive = false; cancelAnimationFrame(rafRef.current); };
  }, []);

  async function startCam() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    videoRef.current.srcObject = stream;
    await videoRef.current.play();
  }

  function packPoint(p) {
    return { x: p?.x ?? 0, y: p?.y ?? 0, confidence: p?.visibility ?? p?.score ?? 0 };
  }
  function packPoints(list, n) {
    const out = [];
    for (let i = 0; i < n; i++) out.push(packPoint(list?.[i]));
    return out;
  }

  async function loop() {
    const v = videoRef.current;
    if (!v) return;

    const nowMs = performance.now();
    const { hand, pose, face } = landmarkersRef.current;

    const handRes = hand.detectForVideo(v, nowMs);
    const poseRes = pose.detectForVideo(v, nowMs);
    const faceRes = face.detectForVideo(v, nowMs);

    let leftLm = null, rightLm = null;
    if (handRes.landmarks?.length) {
      for (let i = 0; i < handRes.landmarks.length; i++) {
        const handed = handRes.handedness?.[i]?.[0]?.categoryName; // Left/Right
        if (handed === "Left") leftLm = handRes.landmarks[i];
        else if (handed === "Right") rightLm = handRes.landmarks[i];
      }
    }

    const frame = {
      pose: packPoints(poseRes.landmarks?.[0], 25),
      leftHand: packPoints(leftLm, 21),
      rightHand: packPoints(rightLm, 21),
      face: packPoints(faceRes.faceLandmarks?.[0], 70),
    };

    framesRef.current.push(frame);
    if (framesRef.current.length > WINDOW) framesRef.current.shift();

    if (framesRef.current.length === WINDOW) {
      const t = Date.now();
      if (t - lastSendRef.current > 250) {
        lastSendRef.current = t;
        try {
          const res = await axios.post("http://localhost:8080/api/translate", { frames: framesRef.current });
          setResult(res.data);
          setErr("");
        } catch (e) {
          setErr(e?.response?.data ? JSON.stringify(e.response.data) : String(e));
        }
      }
    }

    rafRef.current = requestAnimationFrame(loop);
  }

  async function onStart() {
    setErr("");
    if (!ready) return;
    await startCam();
    framesRef.current = [];
    setRunning(true);
    rafRef.current = requestAnimationFrame(loop);
  }

  function onStop() {
    setRunning(false);
    cancelAnimationFrame(rafRef.current);
    const v = videoRef.current;
    if (v?.srcObject) {
      v.srcObject.getTracks().forEach(t => t.stop());
      v.srcObject = null;
    }
  }

  return (
    <div style={{ padding: 16, maxWidth: 760, margin: "0 auto" }}>
      <h2>Translator (pose+face+hands, 30 frames)</h2>

      <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
        <button disabled={!ready || running} onClick={onStart}>Start</button>
        <button disabled={!running} onClick={onStop}>Stop</button>
        {!ready && <span>Loading models…</span>}
      </div>

      <div style={{ marginTop: 12 }}>
        <video ref={videoRef} playsInline muted style={{ width: "100%", borderRadius: 12, background: "#000" }} />
      </div>

      {err && <pre style={{ whiteSpace: "pre-wrap", color: "crimson" }}>{err}</pre>}

      {result && (
        <div style={{ marginTop: 12, padding: 12, border: "1px solid #ddd", borderRadius: 12 }}>
          <div><b>text:</b> {result.text}</div>
          <div><b>label:</b> {result.label}</div>
          <div><b>confidence:</b> {Number(result.confidence).toFixed(3)}</div>
          <div style={{ marginTop: 8 }}>
            <b>top5:</b>
            <ul>
              {result.candidates?.map((c, i) => (
                <li key={i}>{c[0]} : {Number(c[1]).toFixed(3)}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
