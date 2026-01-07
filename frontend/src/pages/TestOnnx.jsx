import React, { useEffect, useMemo, useState } from "react";
import * as ort from "onnxruntime-web";

function pretty(obj) {
  try {
    return JSON.stringify(obj, null, 2);
  } catch {
    return String(obj);
  }
}

async function fetchJson(url) {
  const res = await fetch(url, { cache: "no-store" });
  const ct = res.headers.get("content-type") || "";
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Failed to fetch ${url}: ${res.status} ${res.statusText}\n${text.slice(0, 300)}`);
  }
  // HTML이 내려오면 여기서 바로 잡아주기
  if (!ct.includes("application/json") && !ct.includes("text/json")) {
    const text = await res.text().catch(() => "");
    throw new Error(
      `Expected JSON but got content-type="${ct}" from ${url}\n` +
        `First bytes: ${text.slice(0, 60)}`
    );
  }
  return res.json();
}

export default function TestOnnx() {
  const base = useMemo(() => import.meta.env.BASE_URL || "/", []);
  const [status, setStatus] = useState("idle");
  const [log, setLog] = useState([]);
  const [err, setErr] = useState(null);

  const [config, setConfig] = useState(null);
  const [labels, setLabels] = useState(null);
  const [sessionInfo, setSessionInfo] = useState(null);
  const [lastOutput, setLastOutput] = useState(null);

  const pushLog = (msg) =>
    setLog((p) => [...p, `[${new Date().toLocaleTimeString()}] ${msg}`]);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        setErr(null);
        setStatus("boot");

        pushLog("boot: configure onnxruntime-web wasm paths (3-file threaded build)");

        // ====== 1) WASM 경로 매핑 (네 public/onnxruntime-web/ 안의 3개) ======
        ort.env.wasm.wasmPaths = {
          "ort-wasm-simd-threaded.wasm": `${base}onnxruntime-web/ort-wasm-simd-threaded.wasm`,
          "ort-wasm-simd-threaded.jsep.wasm": `${base}onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm`,
          "ort-wasm-simd-threaded.asyncify.wasm": `${base}onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm`,
        };

        // COOP/COEP 없이도 최대한 덜 터지게 1로 고정
        ort.env.wasm.numThreads = 1;

        // 디버그
        ort.env.logLevel = "verbose";
        ort.env.debug = true;

        // ====== 2) artifacts_2h 에서 config/labels 로드 ======
        setStatus("load-json");
        const CFG_URL = `${base}artifacts_2h/attn_config.json`;
        const LABELS_URL = `${base}artifacts_2h/labels.json`; // 필요시 labels_ko.json으로 변경

        pushLog(`load: ${CFG_URL}`);
        pushLog(`load: ${LABELS_URL}`);

        const [cfg, lbl] = await Promise.all([fetchJson(CFG_URL), fetchJson(LABELS_URL)]);

        if (cancelled) return;

        setConfig(cfg);
        setLabels(lbl);

        // ====== 3) config에서 T / input_dim 뽑기 ======
        const T =
          Number(cfg?.T ?? cfg?.t ?? cfg?.seq_len ?? cfg?.time_steps ?? cfg?.window ?? cfg?.frames);
        const inputDim = Number(cfg?.input_dim ?? cfg?.inputDim ?? cfg?.feature_dim ?? cfg?.D);

        if (!Number.isFinite(T) || !Number.isFinite(inputDim)) {
          throw new Error(
            `attn_config.json에서 T/input_dim을 못 찾음.\n` +
              `keys=${Object.keys(cfg || {}).join(", ")}\n` +
              `T=${cfg?.T} input_dim=${cfg?.input_dim}`
          );
        }

        pushLog(`config: T=${T}, input_dim=${inputDim}`);

        // ====== 4) 세션 생성 (네 파일명: attn_lstm.onnx) ======
        setStatus("create-session");
        const ONNX_URL = `${base}artifacts_2h/attn_lstm.onnx`;

        pushLog(`session: creating from ${ONNX_URL}`);

        const session = await ort.InferenceSession.create(ONNX_URL, {
          executionProviders: ["wasm"],
          graphOptimizationLevel: "all",
        });

        if (cancelled) return;

        const inputName = session.inputNames?.[0];
        const outputNames = session.outputNames || [];
        if (!inputName) throw new Error("session.inputNames[0] 없음 (모델 입력 이름 확인 필요)");

        setSessionInfo({
          inputNames: session.inputNames,
          outputNames: session.outputNames,
        });

        pushLog(`session: created. input=${inputName}, outputs=${outputNames.join(", ")}`);

        // ====== 5) 더미 입력으로 1회 run (shape: [1, T, inputDim]) ======
        setStatus("run-infer");
        pushLog("infer: run once with dummy zero tensor [1,T,inputDim]");

        const data = new Float32Array(1 * T * inputDim); // zeros
        const inputTensor = new ort.Tensor("float32", data, [1, T, inputDim]);

        const results = await session.run({ [inputName]: inputTensor });

        if (cancelled) return;

        const firstOutName = outputNames[0] || Object.keys(results)[0];
        const outTensor = results[firstOutName];

        setLastOutput({
          picked: firstOutName,
          dtype: outTensor?.type,
          dims: outTensor?.dims,
          sample: outTensor?.data ? Array.from(outTensor.data).slice(0, 20) : null,
        });

        pushLog(
          `infer: done. picked=${firstOutName} dims=${outTensor?.dims?.join("x") || "?"}`
        );

        setStatus("done");
      } catch (e) {
        if (cancelled) return;
        console.error(e);
        const msg = String(e?.stack || e?.message || e);
        setErr(msg);
        setStatus("error");
        pushLog(`ERROR: ${String(e?.message || e)}`);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [base]);

  return (
    <div style={{ padding: 16, fontFamily: "ui-sans-serif, system-ui", lineHeight: 1.45 }}>
      <h2>TestOnnx</h2>

      <div style={{ marginBottom: 12 }}>
        <b>Status:</b> {status}
      </div>

      {err && (
        <div style={{ whiteSpace: "pre-wrap", padding: 12, border: "1px solid #f00", marginBottom: 12 }}>
          <b>Error</b>
          <div>{err}</div>
        </div>
      )}

      <div style={{ display: "grid", gap: 12, gridTemplateColumns: "1fr 1fr" }}>
        <section style={{ border: "1px solid #ddd", padding: 12 }}>
          <b>Session Info</b>
          <pre style={{ whiteSpace: "pre-wrap" }}>{pretty(sessionInfo)}</pre>
        </section>

        <section style={{ border: "1px solid #ddd", padding: 12 }}>
          <b>Last Output (sample)</b>
          <pre style={{ whiteSpace: "pre-wrap" }}>{pretty(lastOutput)}</pre>
        </section>

        <section style={{ border: "1px solid #ddd", padding: 12 }}>
          <b>attn_config.json</b>
          <pre style={{ whiteSpace: "pre-wrap", maxHeight: 240, overflow: "auto" }}>
            {pretty(config)}
          </pre>
        </section>

        <section style={{ border: "1px solid #ddd", padding: 12 }}>
          <b>labels.json</b>
          <pre style={{ whiteSpace: "pre-wrap", maxHeight: 240, overflow: "auto" }}>
            {pretty(labels)}
          </pre>
        </section>
      </div>

      <section style={{ border: "1px solid #ddd", padding: 12, marginTop: 12 }}>
        <b>Log</b>
        <pre style={{ whiteSpace: "pre-wrap", maxHeight: 280, overflow: "auto" }}>
          {log.join("\n")}
        </pre>
      </section>
    </div>
  );
}
