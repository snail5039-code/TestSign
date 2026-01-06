import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8080",
        changeOrigin: true,
        secure: false,  // HTTPS를 사용할 경우에는 true로 설정, HTTP면 false
        logLevel: "debug",  // 디버깅 로그를 출력하여 요청이 제대로 가고 있는지 확인
      },
    },
  },
});

