import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const API_URL = "http://127.0.0.1:8080/api/translate";

// ===== 모델 입력 고정 =====
const TARGET_FRAMES = 30;

// ===== 캡처 정책 =====
// 작은 동작 살리려면 raw를 고FPS로 모으고, 끝에 30프레임으로 리샘플.
const CAPTURE_MS = 3000;

// 너무 고FPS로 저장하면 브라우저 부담될 수 있어서 상한(권장 30~60)
// 0이면 rAF 그대로(보통 60fps)
const RAW_FPS_CAP = 60;
const RAW_INTERVAL_MS = RAW_FPS_CAP > 0 ? 1000 / RAW_FPS_CAP : 0;

// ===== fixed sizes: pose75 + face210 + left63 + right63 = 411 =====
const ZERO75 = Array(25 * 3).fill(0);
const ZERO63 = Array(21 * 3).fill(0);
const ZERO210 = Array(70 * 3).fill(0);

// ---------- helpers ----------
function isFiniteNumber(x) {
  return typeof x === "number" && Number.isFinite(x);
}

// NOTE:
// 학습 데이터가 (x,y,visibility/presence)였으면 이대로.
// 학습이 (x,y,z)였다면 c 대신 lm.z를 넣어야 분포가 맞음.
function toFlatXYZNormalized(landmarks, confDefault = 1) {
  const out = [];
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    const x = lm?.x ?? 0;
    const y = lm?.y ?? 0;
    const c = (lm?.visibility ?? lm?.presence ?? confDefault ?? 1) || 1;
    out.push(x, y, c);
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

function concatStats(frames) {
  if (!frames?.length) return { min: 0, max: 0, mean: 0 };
  let min = Infinity,
    max = -Infinity,
    sum = 0,
    cnt = 0;

  for (const fr of frames) {
    const arr = []
      .concat(fr.pose || [], fr.face || [], fr.leftHand || [], fr.rightHand || []);
    for (const v of arr) {
      if (!isFiniteNumber(v)) continue;
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
      cnt += 1;
    }
  }
  const mean = cnt ? sum / cnt : 0;
  return { min: Number.isFinite(min) ? min : 0, max: Number.isFinite(max) ? max : 0, mean };
}

function hasAnyNonZero(arr) {
  if (!Array.isArray(arr) || arr.length === 0) return false;
  for (const v of arr) if (isFiniteNumber(v) && v !== 0) return true;
  return false;
}

/**
 * 시간축 기준 리샘플
 * - frames: [{t, pose, face, leftHand, rightHand}, ...] (t는 ms)
 * - startT~endT를 TARGET_FRAMES개로 균등 분할하고, 각 시점에 가장 가까운 프레임을 선택
 */
function resampleByTime(frames, startT, endT, targetN) {
  if (!frames?.length) {
    // fallback: 전부 0
    return Array.from({ length: targetN }, () => ({
      pose: [...ZERO75],
      face: [...ZERO210],
      leftHand: [...ZERO63],
      rightHand: [...ZERO63],
    }));
  }

  const src = frames.slice().sort((a, b) => (a.t ?? 0) - (b.t ?? 0));
  const out = [];

  const span = Math.max(1, endT - startT);
  let j = 0;

  for (let i = 0; i < targetN; i++) {
    const targetT = startT + (span * i) / (targetN - 1);

    while (j + 1 < src.length && src[j + 1].t <= targetT) j++;

    // j 또는 j+1 중 더 가까운 것 선택
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

// ======= Landmark drawing =======
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

function drawLandmarks(ctx, pts, w, h, connections) {
  if (!pts || pts.length === 0) return;

  // lines
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(0,255,0,0.85)";
  ctx.beginPath();
  for (const [a, b] of connections) {
    const pa = pts[a], pb = pts[b];
    if (!pa || !pb) continue;
    ctx.moveTo(pa.x * w, pa.y * h);
    ctx.lineTo(pb.x * w, pb.y * h);
  }
  ctx.stroke();

  // points
  ctx.fillStyle = "rgba(255,0,0,0.9)";
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    if (!p) continue;
    const x = p.x * w, y = p.y * h;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

export default function TranslatorCam() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const rafRef = useRef(null);

  const [ready, setReady] = useState(false);
  const [camOn, setCamOn] = useState(false);

  // UI 상태
  const [armedUI, setArmedUI] = useState(false);
  const [recordingUI, setRecordingUI] = useState(false);
  const [captured, setCaptured] = useState(false);

  const [framesCount, setFramesCount] = useState(0); // 최종 30 기준 카운트(완료 시 30)
  const [rawCount, setRawCount] = useState(0);       // raw로 몇 개 모였는지
  const [msLeft, setMsLeft] = useState(0);

  const [dbg, setDbg] = useState({ min: 0, max: 0, mean: 0, handCount: 0 });

  const [label, setLabel] = useState("-");
  const [text, setText] = useState("-");
  const [confidence, setConfidence] = useState(0);
  const [top5UI, setTop5UI] = useState([]);

  // buffers
  const rawFramesRef = useRef([]);     // 고FPS 원본
  const finalFramesRef = useRef([]);   // 리샘플 30프레임
  const inFlightRef = useRef(false);

  // mediapipe
  const handRef = useRef(null);
  const poseRef = useRef(null);

  // loop flags
  const runningRef = useRef(false);

  // closure-proof state refs
  const armedRef = useRef(false);
  const recordingRef = useRef(false);

  // capture timing
  const captureStartRef = useRef(0);
  const captureEndRef = useRef(0);
  const lastRawPushRef = useRef(0);

  // hold-last hands (짧은 미검출 구간 완화)
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

    // resize canvas
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.width = video.videoWidth || 560;
      canvas.height = video.videoHeight || 420;
    }

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

    captureStartRef.current = 0;
    captureEndRef.current = 0;
    lastRawPushRef.current = 0;

    runningRef.current = true;
    setCamOn(true);

    const loop = () => {
      if (!runningRef.current) return;
      rafRef.current = requestAnimationFrame(loop);

      const hand = handRef.current;
      const pose = poseRef.current;
      if (!hand || !pose) return;
      if (video.readyState < 2) return;

      const now = performance.now();

      // detect
      const handRes = hand.detectForVideo(video, now);
      const poseRes = pose.detectForVideo(video, now);

      // draw overlay
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        const w = canvas.width, h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        const posePts = poseRes?.landmarks?.[0] || [];
        drawLandmarks(ctx, posePts, w, h, POSE_CONNECTIONS);

        const handPtsList = handRes?.landmarks || [];
        for (const pts of handPtsList) drawLandmarks(ctx, pts, w, h, HAND_CONNECTIONS);
      }

      // hands parsing
      let left = null;
      let right = null;
      const handCount = handRes?.landmarks?.length ? handRes.landmarks.length : 0;
      const handDetected = handCount > 0;

      if (handDetected) {
        for (let i = 0; i < handRes.landmarks.length; i++) {
          const lm = handRes.landmarks[i];
          const handed = handRes.handednesses?.[i]?.[0]?.categoryName; // "Left"/"Right"
          const flat = toFlatXYZNormalized(lm, 1);

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

      const handActive = hasAnyNonZero(leftFinal) || hasAnyNonZero(rightFinal);

      // pose
      let poseFlat = ZERO75;
      if (poseRes?.landmarks?.[0]?.length) {
        const p = poseRes.landmarks[0].slice(0, 25);
        poseFlat = toFlatXYZNormalized(p, 1);
      }
      const faceFlat = ZERO210;

      // 1) armed 상태에서 손이 처음 잡히는 순간 capture 시작
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

      // 2) recording 중이면 raw를 고FPS로 쌓기 (손이 잠깐 사라져도 hold-last로 값 유지)
      if (recordingRef.current) {
        const endT = captureEndRef.current;
        const leftMs = Math.max(0, endT - now);
        setMsLeft(Math.ceil(leftMs));

        // FPS cap (옵션)
        if (RAW_INTERVAL_MS > 0 && now - lastRawPushRef.current < RAW_INTERVAL_MS) {
          // skip pushing but continue loop
        } else {
          lastRawPushRef.current = now;

          const frame = {
            t: now,
            pose: [...poseFlat],
            face: [...faceFlat],
            leftHand: [...leftFinal],
            rightHand: [...rightFinal],
          };

          rawFramesRef.current.push(frame);
          setRawCount(rawFramesRef.current.length);
        }

        // capture 종료
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

      // debug (raw 기준이 아니라 "현재 마지막 프레임/버퍼" 기준으로 보려면 raw를 쓰는 게 직관적)
      const { min, max, mean } = concatStats(
        finalFramesRef.current.length ? finalFramesRef.current : (rawFramesRef.current.slice(-30) || [])
      );
      setDbg({ min, max, mean, handCount });
    };

    rafRef.current = requestAnimationFrame(loop);
  }

  function stopCam() {
    runningRef.current = false;
    setCamOn(false);

    armedRef.current = false;
    recordingRef.current = false;
    setArmedUI(false);
    setRecordingUI(false);
    setMsLeft(0);

    if (rafRef.current) cancelAnimationFrame(rafRef.current);

    const video = videoRef.current;
    if (video?.srcObject) {
      const tracks = video.srcObject.getTracks();
      tracks.forEach((t) => t.stop());
      video.srcObject = null;
    }

    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }

  function record() {
    if (!camOn) return;
    if (armedRef.current || recordingRef.current) return;

    // “예전 손”이 시작 조건 통과시키지 않게 reset
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

    setLabel("-");
    setText("-");
    setConfidence(0);
    setTop5UI([]);
  }

  async function sendPredict() {
    if (!captured) return;
    if (inFlightRef.current) return;

    const frames = deepCloneFrames(finalFramesRef.current);
    if (frames.length !== TARGET_FRAMES) {
      console.warn(`[SEND] frames=${frames.length} (need ${TARGET_FRAMES})`);
      return;
    }

    inFlightRef.current = true;

    try {
      const first = frames[0];
      console.log("[PREDICT] frames len=", frames.length);
      console.log(
        "[PREDICT] lens pose/face/left/right =",
        first?.pose?.length,
        first?.face?.length,
        first?.leftHand?.length,
        first?.rightHand?.length
      );
      console.log("[PREDICT] left first6 =", first?.leftHand?.slice(0, 6));
      console.log("[PREDICT] right first6 =", first?.rightHand?.slice(0, 6));

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

      console.log("[PREDICT] response =", data);
    } catch (e) {
      console.error("Error during predict:", e);
    } finally {
      inFlightRef.current = false;
    }
  }

  const timeProgress =
    recordingUI && captureStartRef.current > 0
      ? Math.min(
          100,
          Math.round(((CAPTURE_MS - msLeft) / CAPTURE_MS) * 100)
        )
      : captured
      ? 100
      : 0;

  return (
    <div style={{ padding: 16 }}>
      <h2>TranslatorCam (고FPS 캡처 → 30프레임 시간리샘플)</h2>

      <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
        <div>
          <div style={{ position: "relative", width: 560 }}>
            <video
              ref={videoRef}
              playsInline
              muted
              style={{ width: 560, borderRadius: 12, background: "#111" }}
            />
            <canvas
              ref={canvasRef}
              style={{
                position: "absolute",
                left: 0,
                top: 0,
                width: 560,
                height: "auto",
                pointerEvents: "none",
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
              Record ({CAPTURE_MS / 1000}s)
            </button>
            <button disabled={!captured || recordingUI || armedUI} onClick={sendPredict}>
              Send (Translate)
            </button>
          </div>

          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: 13, opacity: 0.85 }}>
              status: {camOn ? "camOn" : "camOff"} | armed(wait hand): {String(armedUI)} | recording:{" "}
              {String(recordingUI)} | captured: {String(captured)}
              <br />
              captureLeft(ms): {recordingUI ? msLeft : 0} | rawFrames: {rawCount} | finalFrames:{" "}
              {framesCount}/{TARGET_FRAMES} | rawFpsCap: {RAW_FPS_CAP}
            </div>

            <div
              style={{
                height: 10,
                background: "#eee",
                borderRadius: 999,
                overflow: "hidden",
                marginTop: 6,
              }}
            >
              <div style={{ width: `${timeProgress}%`, height: "100%", background: "#111" }} />
            </div>

            <div style={{ marginTop: 10, fontSize: 12, opacity: 0.75 }}>
              debug: min={dbg.min.toFixed(4)} max={dbg.max.toFixed(4)} mean={dbg.mean.toFixed(4)} | handCount={dbg.handCount}
              <br />
              ※ Record 후 손이 처음 잡히는 순간부터 {CAPTURE_MS / 1000}s 동안 raw를 촘촘히 모으고,
              <br />
              ※ 시간축 기준으로 30프레임으로 리샘플해서 모델에 넣습니다(작은 동작 보존).
            </div>
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
