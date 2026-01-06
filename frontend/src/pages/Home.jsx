import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-white">
      <div className="mx-auto max-w-3xl px-4 py-10">
        {/* 헤더 */}
        <div className="mb-6">
          <h1 className="text-3xl font-black tracking-tight text-slate-900">
            메인
          </h1>
          <p className="mt-2 text-sm text-slate-600">
            원하는 기능 페이지로 이동해보자.
          </p>
        </div>

        {/* 카드 */}
        <div className="rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-200">
          <div className="mb-3 text-xs font-semibold text-slate-500">
            빠른 이동
          </div>

          <ul className="space-y-3">
            <li>
              <Link
                to="/translatorCam"
                className="flex items-center justify-between rounded-2xl bg-slate-50 px-4 py-4 ring-1 ring-slate-200 hover:bg-slate-100"
              >
                <span className="font-semibold text-slate-900">
                  이게 진짜임
                </span>
                <span className="text-slate-400">→</span>
              </Link>
            </li>
            <li>
              <Link
                to="/collect"
                className="flex items-center justify-between rounded-2xl bg-slate-50 px-4 py-4 ring-1 ring-slate-200 hover:bg-slate-100"
              >
                <span className="font-semibold text-slate-900">
                  영상 추출
                </span>
                <span className="text-slate-400">→</span>
              </Link>
            </li>

            <li>
              <Link
                to="/camera"
                className="flex items-center justify-between rounded-2xl bg-slate-900 px-4 py-4 text-white hover:bg-slate-800"
              >
                <span className="font-semibold">카메라</span>
                <span className="text-white/70">→</span>
              </Link>
            </li>

            <li>
              <Link
                to="/callLobby"
                className="flex items-center justify-between rounded-2xl bg-slate-900 px-4 py-4 text-white hover:bg-slate-800"
              >
                <span className="font-semibold">실시간 영상 통화</span>
                <span className="text-white/70">→</span>
              </Link>
            </li>

            <li>
              <Link
                to="/translationLogPanel"
                className="flex items-center justify-between rounded-2xl bg-slate-50 px-4 py-4 ring-1 ring-slate-200 hover:bg-slate-100"
              >
                <span className="font-semibold text-slate-900">
                  번역 로그 (개발참고)
                </span>
                <span className="text-slate-400">→</span>
              </Link>
            </li>
            <li>
              <Link to="/translator"
                className="flex items-center justify-between rounded-2xl bg-slate-900 px-4 py-4 text-white hover:bg-slate-800"
              >
                <span className="font-semibold">손 번역(TranslatorCam)</span>
                <span className="text-white/70">→</span>
              </Link>
            </li>

          </ul>

          <div className="mt-4 text-xs text-slate-500">
            * 나중에 로그인/설정 같은 메뉴도 여기 추가하면 됨
          </div>
        </div>

        {/* 하단 작은 안내 */}
        <div className="mt-6 text-xs text-slate-400">
          SLT 프로젝트 · React + Tailwind
        </div>
      </div>
    </div>
  );
}
