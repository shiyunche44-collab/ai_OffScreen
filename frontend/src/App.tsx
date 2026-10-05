import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Jobs } from "./pages/Jobs";
import { Library } from "./pages/Library";
import { NotFound } from "./pages/NotFound";
import { ProjectPage } from "./pages/Project";

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Navigate to="/library" replace />} />
        <Route path="library" element={<Library />} />
        <Route path="jobs" element={<Jobs />} />
        <Route path="projects/:projectId" element={<ProjectPage />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
