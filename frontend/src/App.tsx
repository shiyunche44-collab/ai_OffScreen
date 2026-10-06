import { Navigate, Route, Routes } from "react-router-dom";
import { AnalysisPage } from "./pages/Analysis";
import { Layout } from "./components/Layout";
import { CutsPage } from "./pages/Cuts";
import { Jobs } from "./pages/Jobs";
import { Library } from "./pages/Library";
import { NotFound } from "./pages/NotFound";
import { ProjectPage } from "./pages/Project";
import { ScriptEditorPage } from "./pages/ScriptEditor";
import { Projects } from "./pages/Projects";

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Navigate to="/library" replace />} />
        <Route path="library" element={<Library />} />
        <Route path="library/:assetId" element={<AnalysisPage />} />
        <Route path="library/:assetId/cuts" element={<CutsPage />} />
        <Route path="jobs" element={<Jobs />} />
        <Route path="projects" element={<Projects />} />
        <Route path="projects/:projectId" element={<ProjectPage />} />
        <Route path="projects/:projectId/script" element={<ScriptEditorPage />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
