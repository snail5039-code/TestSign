// 디비에서 보기 귀찮아서 만든거임 그래서 그냥 복붙함ㅋㅋ

import { useEffect, useMemo, useState } from "react";
import axios from "axios";

/**
 * 번역 로그 패널
 * - Spring: GET /api/translation-log?limit=10
 * - 응답 예: [{ id, createdAt, text, confidence, label?, mode? }, ...]
 *
 * 포인트:
 * 1) "번역 실패" / null / confidence 낮음은 실패로 표시
 * 2) 자동 갱신 on/off 가능
 * 3) Tailwind로 화면 통일
 */
export default function TranslationLogPanel({
  limit = 10,
  autoRefreshDefault = true,
  intervalMs = 2000,
}) {
  const [logs, setLogs] = useState([]);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(autoRefreshDefault);

  // createdAt 포맷: DB가 문자열로 주든 ISO로 주든 최대한 처리
  const fmt = (createdAt) => {
    if (!createdAt) return "";
    const d = new Date(createdAt);
    if (Number.isNaN(d.getTime())) return String(createdAt);
    return new Intl.DateTimeFormat("ko-KR", {
      dateStyle: "short",
      timeStyle: "medium",
    }).format(d);
  };

  const fetchLogs = async () => {
    try {
      setLoading(true);
      setErr("");
      const res = await axios.get("/api/translation-log", { params: { limit } });
      const data = res.data;
      setLogs(Array.isArray(data) ? data : []);
    } catch (e) {
      setLogs([]);
      setErr("로그 조회 실패");
      console.log(e);
    } finally {
      setLoading(false);
    }
  };

  // 실패 판정 기준: 네 화면이랑 최대한 자연스럽게
  const normalized = useMemo(() => {
    return (logs ?? []).map((x) => {
      const text = x?.text ?? null;
      const conf = Number(x?.confidence ?? 0);
      const label = x?.label ?? null; // 있으면 보여주기용
      const mode = x?.mode ?? null; // 있으면 보여주기용

      // 실패 기준(원하면 여기만 바꾸면 됨)
      const isFail =
        !text ||
        text === "번역 실패" ||
        Number.isNaN(conf) ||
        conf <= 0.0;

      return {
        ...x,
        text,
        conf: Number.isNaN(conf) ? 0 : conf,
        label,
        mode,
        isFail,
      };
    });
  }, [logs]);

  useEffect(() => {
    fetchLogs();
  }, [limit]);

  useEffect(() => {
    if (!autoRefresh) return;
    const t = setInterval(fetchLogs, intervalMs);
    return () => clearInterval(t);
  }, [autoRefresh, intervalMs, limit]);

  return (
    <div className="w-full max-w-4xl mx-auto">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-lg font-bold">번역 로그</h2>

        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 text-sm text-slate-600">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            자동갱신
          </label>

          <button
            onClick={fetchLogs}
            className="rounded-lg border px-3 py-1.5 text-sm hover:bg-slate-50"
          >
            새로고침
          </button>
        </div>
      </div>

      {err && (
        <div className="mb-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {err}
        </div>
      )}

      <div className="rounded-2xl border bg-white p-3">
        {loading && normalized.length === 0 ? (
          <div className="py-6 text-center text-slate-500 text-sm">불러오는 중...</div>
        ) : normalized.length === 0 ? (
          <div className="py-6 text-center text-slate-500 text-sm">아직 로그가 없어요</div>
        ) : (
          <div className="flex flex-col gap-2">
            {normalized.map((x) => (
              <div
                key={x.id}
                className="rounded-xl border px-4 py-3 flex items-start justify-between"
              >
                <div>
                  <div
                    className={`font-semibold ${
                      x.isFail ? "text-red-600" : "text-green-700"
                    }`}
                  >
                    {x.isFail ? "❌ 번역 실패" : `✅ ${x.text}`}
                  </div>

                  {/* label/mode는 있으면 보이게만(없으면 자동으로 안 보임) */}
                  <div className="mt-1 text-xs text-slate-600">
                    conf: {x.conf.toFixed(2)} · {fmt(x.createdAt)} · #{x.id}
                    {x.label ? ` · ${x.label}` : ""}
                    {x.mode ? ` · mode=${x.mode}` : ""}
                  </div>
                </div>

                {/* 오른쪽 상태 뱃지 */}
                <div
                  className={`text-xs px-2 py-1 rounded-full border ${
                    x.isFail
                      ? "border-red-200 bg-red-50 text-red-700"
                      : "border-green-200 bg-green-50 text-green-800"
                  }`}
                >
                  {x.isFail ? "FAIL" : "OK"}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
