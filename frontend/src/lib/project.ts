import type { Job, ProjectDetail, ProjectOptions } from "../api/types";
import type { StepStage } from "../api/queries";

export const STEPS: { stage: StepStage; label: string; action: string }[] = [
  { stage: "analysis.story", label: "分析影片", action: "分析" },
  { stage: "creation.script", label: "解说文案", action: "生成文案" },
  { stage: "creation.plan", label: "剪辑计划与配音", action: "构建计划" },
  { stage: "output.render", label: "渲染成片", action: "渲染" },
];

export type StepState =
  | { kind: "done" }
  | { kind: "queued" | "running"; job: Job }
  | { kind: "failed"; job: Job }
  | { kind: "none" };

const DEFAULTS = { minutes: 3, voice: null, style: "suspense", spoil_ending: true };

/** The project's options with the schema defaults filled in. */
export function optionsOf(detail: ProjectDetail): Required<Pick<ProjectOptions, "minutes" | "style" | "spoil_ending">> & {
  voice: string | null;
} {
  const o: Partial<ProjectOptions> = detail.project.options ?? {};
  return {
    minutes: o.minutes ?? DEFAULTS.minutes,
    voice: o.voice ?? DEFAULTS.voice,
    style: o.style ?? DEFAULTS.style,
    spoil_ending: o.spoil_ending ?? DEFAULTS.spoil_ending,
  };
}

function sameOptions(scope: unknown, detail: ProjectDetail): boolean {
  const mine = optionsOf(detail);
  const theirs = (scope ?? {}) as Record<string, unknown>;
  return (
    theirs["minutes"] === mine.minutes &&
    (theirs["voice"] ?? null) === mine.voice &&
    theirs["style"] === mine.style &&
    theirs["spoil_ending"] === mine.spoil_ending
  );
}

const newest = (a: Job, b: Job) =>
  a.created_at === b.created_at ? b.id.localeCompare(a.id) : b.created_at.localeCompare(a.created_at);

/**
 * Where one step of a project stands. Built when its stage is cached; otherwise taken from the
 * live jobs of the same asset. Creative steps only count jobs run with this project's options
 * (another project of the same film may ask for a different length); analysis ignores options.
 */
export function stepState(stage: StepStage, detail: ProjectDetail, jobs: Job[] | undefined): StepState {
  if (detail.stages.find((s) => s.stage === stage)?.cached) return { kind: "done" };
  const mine = (jobs ?? [])
    .filter(
      (j) =>
        j.stage === stage &&
        j.scope["asset_id"] === detail.project.asset_id &&
        (stage === "analysis.story" || sameOptions(j.scope["options"], detail)),
    )
    .sort(newest);
  const active = mine.find((j) => j.status === "running" || j.status === "queued");
  if (active) return { kind: active.status === "running" ? "running" : "queued", job: active };
  const latest = mine[0];
  if (latest?.status === "failed") return { kind: "failed", job: latest };
  return { kind: "none" };
}

/** `/api/files/...` URL of a file under the data directory. */
export function fileUrl(relativePath: string): string {
  return `/api/files/${relativePath.split("/").map(encodeURIComponent).join("/")}`;
}

/** Narration speed the script writer aims for (characters per second). */
export const CHARS_PER_SECOND = 4.5;
