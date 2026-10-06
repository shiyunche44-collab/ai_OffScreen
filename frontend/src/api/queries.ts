import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, unwrap } from "./client";
import { mergeJobs } from "./events";
import type { CharacterEdit, CharactersView, CutsView, Job } from "./types";

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

export function useAsset(assetId: string) {
  return useQuery({
    queryKey: keys.asset(assetId),
    queryFn: async () =>
      unwrap(await api.GET("/api/assets/{asset_id}", { params: { path: { asset_id: assetId } } })),
  });
}

/** One part of an asset's MovieIndex. Each is its own request, so a stage that is not built
 * yet (404) leaves the other panels working. */
export const indexKey = (assetId: string, part: string) => [...keys.asset(assetId), "index", part] as const;

export function useShots(assetId: string) {
  return useQuery({
    queryKey: indexKey(assetId, "shots"),
    queryFn: async () =>
      unwrap(await api.GET("/api/assets/{asset_id}/index/shots", { params: { path: { asset_id: assetId } } })),
  });
}

export function useTranscript(assetId: string) {
  return useQuery({
    queryKey: indexKey(assetId, "transcript"),
    queryFn: async () =>
      unwrap(await api.GET("/api/assets/{asset_id}/index/transcript", { params: { path: { asset_id: assetId } } })),
  });
}

export function useScenes(assetId: string) {
  return useQuery({
    queryKey: indexKey(assetId, "scenes"),
    queryFn: async () =>
      unwrap(await api.GET("/api/assets/{asset_id}/index/scenes", { params: { path: { asset_id: assetId } } })),
  });
}

export function useStory(assetId: string) {
  return useQuery({
    queryKey: indexKey(assetId, "story"),
    queryFn: async () =>
      unwrap(await api.GET("/api/assets/{asset_id}/index/story", { params: { path: { asset_id: assetId } } })),
  });
}

export function useCharacters(assetId: string) {
  return useQuery({
    queryKey: indexKey(assetId, "characters"),
    queryFn: async () =>
      unwrap(await api.GET("/api/assets/{asset_id}/index/characters", { params: { path: { asset_id: assetId } } })),
    retry: false, // a 404 just means "not built yet"
  });
}

/** Queue face detection, grouping and naming; the job then shows up through /api/events. */
export function useBuildCharacters(assetId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (): Promise<Job> =>
      unwrap(await api.POST("/api/assets/{asset_id}/characters:build", { params: { path: { asset_id: assetId } } })),
    onSuccess: (job) => client.setQueryData<Job[]>(keys.jobs, (jobs) => mergeJobs(jobs, [job])),
  });
}

/** Rename, ignore, merge or reset one character; the answer is the characters as they now read. */
export function useEditCharacter(assetId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async ({ id, change }: { id: string; change: Partial<CharacterEdit> }): Promise<CharactersView> =>
      unwrap(
        await api.PATCH("/api/assets/{asset_id}/characters/{character_id}", {
          params: { path: { asset_id: assetId, character_id: id } },
          body: { reset: false, ...change }, // only the fields present are touched
        }),
      ),
    onSuccess: (view) => client.setQueryData(indexKey(assetId, "characters"), view),
  });
}

export const cutsKey = (assetId: string) => [...keys.asset(assetId), "cuts"] as const;

/** The hard cuts a person has marked in a movie (the ground truth for shot detection). */
export function useCuts(assetId: string) {
  return useQuery({
    queryKey: cutsKey(assetId),
    queryFn: async () =>
      unwrap(await api.GET("/api/assets/{asset_id}/annotations/cuts", { params: { path: { asset_id: assetId } } })),
  });
}

export function useSaveCuts(assetId: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (cuts: number[]): Promise<CutsView> =>
      unwrap(
        await api.PUT("/api/assets/{asset_id}/annotations/cuts", {
          params: { path: { asset_id: assetId } },
          body: { cuts },
        }),
      ),
    onSuccess: (view) => {
      client.setQueryData(cutsKey(assetId), view);
      void client.invalidateQueries({ queryKey: [...cutsKey(assetId), "evaluation"] });
    },
  });
}

/** Precision / recall / F1 of the detected shots against the saved marks. */
export function useCutEvaluation(assetId: string, tolerance: number, enabled: boolean) {
  return useQuery({
    queryKey: [...cutsKey(assetId), "evaluation", tolerance] as const,
    enabled,
    retry: false, // a 404 means the shots are not built yet
    queryFn: async () =>
      unwrap(
        await api.GET("/api/assets/{asset_id}/annotations/cuts/evaluation", {
          params: { path: { asset_id: assetId }, query: { tolerance } },
        }),
      ),
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

export function useProjects() {
  return useQuery({
    queryKey: keys.projects,
    queryFn: async () => unwrap(await api.GET("/api/projects")),
  });
}

/** The generated script; only asked for once the project says it exists. */
export function useScript(projectId: string, enabled: boolean) {
  return useQuery({
    queryKey: [...keys.project(projectId), "script"] as const,
    enabled,
    queryFn: async () =>
      unwrap(await api.GET("/api/projects/{project_id}/script", { params: { path: { project_id: projectId } } })),
  });
}

export type StepStage = "analysis.story" | "creation.script" | "creation.plan" | "output.render";

/** Queue one step of the project; the job then shows up through /api/events. */
export function useRunStep(project: { id: string; asset_id: string }) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (stage: StepStage): Promise<Job> => {
      const path = { path: { project_id: project.id } };
      switch (stage) {
        case "analysis.story":
          return unwrap(
            await api.POST("/api/assets/{asset_id}/analyze", { params: { path: { asset_id: project.asset_id } } }),
          );
        case "creation.script":
          return unwrap(await api.POST("/api/projects/{project_id}/script:generate", { params: path }));
        case "creation.plan":
          return unwrap(await api.POST("/api/projects/{project_id}/plan:build", { params: path }));
        case "output.render":
          return unwrap(await api.POST("/api/projects/{project_id}/render", { params: path }));
      }
    },
    onSuccess: (job) => client.setQueryData<Job[]>(keys.jobs, (jobs) => mergeJobs(jobs, [job])),
  });
}
