import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, unwrap } from "./client";
import { mergeJobs } from "./events";
import type { Job } from "./types";

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

export function useBrowse(path: string | undefined) {
  return useQuery({
    queryKey: ["media", path ?? null] as const,
    queryFn: async () => unwrap(await api.GET("/api/assets/browse", { params: { query: { path } } })),
    staleTime: 0,
  });
}

export function useImportAsset() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (path: string) => unwrap(await api.POST("/api/assets", { body: { path } })),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: keys.assets });
      void client.invalidateQueries({ queryKey: ["media"] }); // the "already imported" marks
    },
  });
}

export function useAnalyzeAsset() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (assetId: string) =>
      unwrap(await api.POST("/api/assets/{asset_id}/analyze", { params: { path: { asset_id: assetId } } })),
    // The job also arrives through /api/events; this makes the button react at once.
    onSuccess: () => void client.invalidateQueries({ queryKey: keys.jobs }),
  });
}

export function useCreateProject() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (assetId: string) =>
      unwrap(await api.POST("/api/projects", { body: { asset_id: assetId } })),
    onSuccess: () => void client.invalidateQueries({ queryKey: keys.projects }),
  });
}

/** The tail of a job's log. Refetched every 2 s while `live` (the job is still running). */
export function useJobLog(jobId: string, live: boolean) {
  return useQuery({
    queryKey: ["job-log", jobId] as const,
    queryFn: async () =>
      unwrap(
        await api.GET("/api/jobs/{job_id}/log", { params: { path: { job_id: jobId } }, parseAs: "text" }),
      ),
    refetchInterval: live ? 2000 : false,
    staleTime: 0,
  });
}

function useJobControl(action: "cancel" | "retry") {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (jobId: string): Promise<Job> => {
      const params = { params: { path: { job_id: jobId } } };
      return unwrap(
        action === "cancel"
          ? await api.POST("/api/jobs/{job_id}:cancel", params)
          : await api.POST("/api/jobs/{job_id}:retry", params),
      );
    },
    // Show the new state at once; /api/events confirms it.
    onSuccess: (job) => client.setQueryData<Job[]>(keys.jobs, (jobs) => mergeJobs(jobs, [job])),
  });
}

export const useCancelJob = () => useJobControl("cancel");
export const useRetryJob = () => useJobControl("retry");
