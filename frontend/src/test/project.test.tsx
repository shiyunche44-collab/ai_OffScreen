import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import type { Job, ProjectDetail } from "../api/types";
import { fileUrl, optionsOf, stepState } from "../lib/project";
import { FakeEventSource, assetDetail, job, json, renderWithProviders, stubApi, stubEventSource } from "./helpers";

beforeEach(() => stubEventSource());
afterEach(() => vi.unstubAllGlobals());

const STAGES = ["analysis.story", "creation.script", "creation.plan", "output.render"];

function detail(opts: { cached?: string[]; video?: string | null; options?: object } = {}): ProjectDetail {
  const { cached = [], video = null, options } = opts;
  return {
    project: {
      id: "prj_1",
      asset_id: "ast_1",
      name: "三分钟版",
      options: (options ?? { minutes: 0.25, voice: null, style: "neutral", spoil_ending: true }) as ProjectDetail["project"]["options"],
      created_at: "2026-10-05T08:00:00Z",
    },
    stages: STAGES.map((stage) => ({ stage, cached: cached.includes(stage) })),
    video,
  };
}

const jobFor = (stage: string, o: Partial<Job> = {}, options: object = { minutes: 0.25, voice: null, style: "neutral", spoil_ending: true }) =>
  job({ id: `job_${stage}`, stage, scope: { asset_id: "ast_1", options }, ...o });

const script = {
  schema_version: 1,
  id: "scr_1",
  project_id: "prj_1",
  version: 1,
  parent_version: null,
  author: "ai",
  params: { style: "neutral", target_duration_s: 15, voice_id: "v" },
  outline: [],
  segments: [
    { id: "seg_01", kind: "narration", beat: "hook", text: "她站在荒原上，喊着那个名字。", scene_refs: ["sc_001"], line_refs: [] },
    { id: "seg_02", kind: "narration", beat: "ending", text: "故事才刚刚开始。", scene_refs: ["sc_002"], line_refs: [] },
  ],
  annotations: [],
};

describe("stepState", () => {
  it("is done when the stage is cached", () => {
    expect(stepState("creation.script", detail({ cached: ["creation.script"] }), [])).toEqual({ kind: "done" });
  });

  it("follows an active job of this project's options", () => {
    const running = jobFor("creation.script", { status: "running", progress: 0.5 });
    expect(stepState("creation.script", detail(), [running])).toMatchObject({ kind: "running" });
    expect(stepState("creation.script", detail(), [jobFor("creation.script", { status: "queued" })]).kind).toBe("queued");
  });

  it("ignores a job of the same film run with other options", () => {
    const other = jobFor("creation.script", { status: "running" }, { minutes: 5, voice: null, style: "neutral", spoil_ending: true });
    expect(stepState("creation.script", detail(), [other]).kind).toBe("none");
  });

  it("does not look at options for analysis", () => {
    const analysis = jobFor("analysis.story", { status: "running" }, { minutes: 3, voice: null, style: "x", spoil_ending: false });
    expect(stepState("analysis.story", detail(), [analysis]).kind).toBe("running");
  });

  it("ignores other assets and shows the latest failure", () => {
    const elsewhere = { ...jobFor("creation.script", { status: "running" }), scope: { asset_id: "ast_2", options: {} } };
    const failed = jobFor("creation.script", { status: "failed", error: "boom" });
    expect(stepState("creation.script", detail(), [elsewhere]).kind).toBe("none");
    expect(stepState("creation.script", detail(), [elsewhere, failed]).kind).toBe("failed");
  });

  it("treats a missing voice like null", () => {
    const j = jobFor("creation.plan", { status: "queued" }, { minutes: 0.25, style: "neutral", spoil_ending: true });
    expect(stepState("creation.plan", detail(), [j]).kind).toBe("queued");
  });
});

describe("helpers", () => {
  it("fills in default options", () => {
    const bare = detail();
    delete (bare.project as { options?: unknown }).options;
    expect(optionsOf(bare)).toEqual({ minutes: 3, voice: null, style: "neutral", spoil_ending: true });
  });

  it("builds file URLs with each path segment encoded", () => {
    expect(fileUrl("artifacts/output.render/ab12/final.mp4")).toBe("/api/files/artifacts/output.render/ab12/final.mp4");
    expect(fileUrl("a b/c#d.mp4")).toBe("/api/files/a%20b/c%23d.mp4");
  });
});

describe("project page", () => {
  const base = { "GET /api/assets": () => json([assetDetail()]) };

  it("shows the project, its options and one row per step", async () => {
    stubApi({ ...base, "GET /api/jobs": () => json([]), "GET /api/projects/prj_1": () => json(detail({ cached: ["analysis.story"] })) });
    renderWithProviders(<App />, "/projects/prj_1");
    expect(await screen.findByRole("heading", { name: "三分钟版" })).toBeInTheDocument();
    expect(screen.getByText(/0\.25 分钟 · 风格 neutral · 音色 默认 · 含结局/)).toBeInTheDocument();
    const steps = within(screen.getByRole("list", { name: "流程" })).getAllByRole("listitem");
    expect(steps.map((s) => s.getAttribute("aria-label"))).toEqual(["分析影片", "解说文案", "剪辑计划与配音", "渲染成片"]);
    const analysis = screen.getByRole("listitem", { name: "分析影片" });
    expect(analysis).toHaveTextContent("已完成");
    expect(within(analysis).queryByRole("button")).not.toBeInTheDocument(); // nothing left to do
    expect(within(screen.getByRole("listitem", { name: "解说文案" })).getByRole("button", { name: "生成文案" })).toBeEnabled();
  });

  it.each([
    ["analysis.story", "分析影片", "分析", "POST /api/assets/ast_1/analyze"],
    ["creation.script", "解说文案", "生成文案", "POST /api/projects/prj_1/script:generate"],
    ["creation.plan", "剪辑计划与配音", "构建计划", "POST /api/projects/prj_1/plan:build"],
    ["output.render", "渲染成片", "渲染", "POST /api/projects/prj_1/render"],
  ])("the %s button queues that step", async (stage, label, action, route) => {
    const post = vi.fn(() => json(jobFor(stage, { status: "queued" }), 202));
    stubApi({ ...base, "GET /api/jobs": () => json([]), "GET /api/projects/prj_1": () => json(detail()), [route]: post });
    renderWithProviders(<App />, "/projects/prj_1");
    const row = await screen.findByRole("listitem", { name: label });
    await userEvent.click(within(row).getByRole("button", { name: action }));
    await waitFor(() => expect(post).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(row).toHaveTextContent("排队中")); // the new job is shown at once
    expect(within(row).getByRole("button")).toBeDisabled();
  });

  it("shows live progress of a running step and blocks a second click", async () => {
    stubApi({
      ...base,
      "GET /api/jobs": () => json([jobFor("creation.script", { status: "running", progress: 0.3 })]),
      "GET /api/projects/prj_1": () => json(detail({ cached: ["analysis.story"] })),
    });
    renderWithProviders(<App />, "/projects/prj_1");
    const row = await screen.findByRole("listitem", { name: "解说文案" });
    await waitFor(() => expect(row).toHaveTextContent("进行中 30%"));
    expect(within(row).getByRole("button")).toBeDisabled();
    FakeEventSource.instances[0]?.emit("job", jobFor("creation.script", { status: "running", progress: 0.9 }));
    await waitFor(() => expect(row).toHaveTextContent("进行中 90%"));
  });

  it("shows a failed step with its error and offers a retry", async () => {
    stubApi({
      ...base,
      "GET /api/jobs": () => json([jobFor("creation.script", { status: "failed", error: "LLMAuthError: bad key" })]),
      "GET /api/projects/prj_1": () => json(detail({ cached: ["analysis.story"] })),
    });
    renderWithProviders(<App />, "/projects/prj_1");
    const row = await screen.findByRole("listitem", { name: "解说文案" });
    await waitFor(() => expect(row).toHaveTextContent("失败：LLMAuthError: bad key"));
    expect(within(row).getByRole("button", { name: "重试" })).toBeEnabled();
  });

  it("shows why a step could not be queued", async () => {
    stubApi({
      ...base,
      "GET /api/jobs": () => json([]),
      "GET /api/projects/prj_1": () => json(detail()),
      "POST /api/projects/prj_1/render": () => json({ error: { code: "invalid_input", message: "minutes must be positive" } }, 422),
    });
    renderWithProviders(<App />, "/projects/prj_1");
    await userEvent.click(within(await screen.findByRole("listitem", { name: "渲染成片" })).getByRole("button"));
    expect(await screen.findByText("minutes must be positive")).toBeInTheDocument();
  });

  it("shows the script once it exists, read-only", async () => {
    stubApi({
      ...base,
      "GET /api/jobs": () => json([]),
      "GET /api/projects/prj_1": () => json(detail({ cached: ["analysis.story", "creation.script"] })),
      "GET /api/projects/prj_1/script": () => json(script),
    });
    renderWithProviders(<App />, "/projects/prj_1");
    const view = await screen.findByLabelText("文案");
    expect(await within(view).findByText("她站在荒原上，喊着那个名字。")).toBeInTheDocument();
    expect(view).toHaveTextContent("共 2 段");
    expect(view).toHaveTextContent("seg_01 · 钩子");
    expect(view).toHaveTextContent("seg_02 · 结尾");
    expect(within(view).queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("does not ask for a script that does not exist", async () => {
    const getScript = vi.fn(() => json({ error: { code: "not_found", message: "no script" } }, 404));
    stubApi({ ...base, "GET /api/jobs": () => json([]), "GET /api/projects/prj_1": () => json(detail()), "GET /api/projects/prj_1/script": getScript });
    renderWithProviders(<App />, "/projects/prj_1");
    await screen.findByRole("heading", { name: "三分钟版" });
    expect(screen.queryByLabelText("文案")).not.toBeInTheDocument();
    expect(getScript).not.toHaveBeenCalled();
  });

  it("plays the finished video from the file service", async () => {
    stubApi({
      ...base,
      "GET /api/jobs": () => json([]),
      "GET /api/projects/prj_1": () => json(detail({ cached: STAGES, video: "artifacts/output.render/ab12/final.mp4" })),
      "GET /api/projects/prj_1/script": () => json(script),
    });
    renderWithProviders(<App />, "/projects/prj_1");
    const video = await screen.findByLabelText("成片");
    expect(video).toHaveAttribute("src", "/api/files/artifacts/output.render/ab12/final.mp4");
    expect(video).toHaveAttribute("controls");
  });

  it("has no player before the render", async () => {
    stubApi({ ...base, "GET /api/jobs": () => json([]), "GET /api/projects/prj_1": () => json(detail()) });
    renderWithProviders(<App />, "/projects/prj_1");
    await screen.findByRole("heading", { name: "三分钟版" });
    expect(screen.queryByLabelText("成片")).not.toBeInTheDocument();
  });

  it("refreshes when a job succeeds, so the script and video appear by themselves", async () => {
    let built = false;
    stubApi({
      ...base,
      "GET /api/jobs": () => json([]),
      "GET /api/projects/prj_1": () => json(built ? detail({ cached: ["analysis.story", "creation.script"] }) : detail({ cached: ["analysis.story"] })),
      "GET /api/projects/prj_1/script": () => json(script),
    });
    renderWithProviders(<App />, "/projects/prj_1");
    await screen.findByRole("heading", { name: "三分钟版" });
    expect(screen.queryByLabelText("文案")).not.toBeInTheDocument();
    built = true;
    FakeEventSource.instances[0]?.emit("job", jobFor("creation.script", { status: "succeeded", progress: 1 }));
    expect(await screen.findByLabelText("文案")).toBeInTheDocument();
  });
});

describe("project list", () => {
  it("lists projects with their film and links to them", async () => {
    stubApi({
      "GET /api/assets": () => json([assetDetail()]),
      "GET /api/projects": () =>
        json([{ id: "prj_1", asset_id: "ast_1", name: "三分钟版", options: {}, created_at: "2026-10-05T08:00:00Z" }]),
      "GET /api/jobs": () => json([]),
    });
    renderWithProviders(<App />, "/projects");
    const link = await screen.findByRole("link", { name: /三分钟版/ });
    expect(link).toHaveAttribute("href", "/projects/prj_1");
    await waitFor(() => expect(link).toHaveTextContent("Sintel"));
  });

  it("explains an empty list and the navigation reaches it", async () => {
    stubApi({ "GET /api/assets": () => json([]), "GET /api/projects": () => json([]), "GET /api/jobs": () => json([]) });
    renderWithProviders(<App />, "/library");
    await userEvent.click(screen.getByRole("link", { name: "项目" }));
    expect(await screen.findByText(/还没有项目/)).toBeInTheDocument();
  });
});
