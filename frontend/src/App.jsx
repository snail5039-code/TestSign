import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home.jsx";
import TranslatorCam from "./pages/TranslatorCam.jsx";
import Collect from "./pages/Collect.jsx";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/Home" element={<Home />} />
      <Route path="/translatorCam" element={<TranslatorCam />} />
      <Route path="/collect" element={<Collect />} />
    </Routes>
  );
}
