import { useQuery } from "@tanstack/react-query";
import { api, unwrap } from "./client";

export const keys = {
  assets: ["assets"] as const,
  asset: (id: string) => ["assets", id] as const,
  projects: ["projects"] as const,
  project: (id: string) => ["projects", id] as const,
  jobs: ["jobs"] as const,
};

export function useAssets() {
  return useQuery({
    queryKey: keys.assets,
    queryFn: async () => unwrap(await api.GET("/api/assets")),
  });
}

export function useJobs() {
  return useQuery({
    queryKey: keys.jobs,
    queryFn: async () => unwrap(await api.GET("/api/jobs")),
  });
}

export function useProject(projectId: string) {
  return useQuery({
    queryKey: keys.project(projectId),
    queryFn: async () =>
      unwrap(await api.GET("/api/projects/{project_id}", { params: { path: { project_id: projectId } } })),
  });
}
