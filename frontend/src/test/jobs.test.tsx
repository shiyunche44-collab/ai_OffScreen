import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { api, unwrap } from "../api/client";
import { filterJobs, runtime, stageLabel } from "../lib/jobs";
import { FakeEventSource, assetDetail, job, json, renderWithProviders, stubApi, stubEventSource } from "./helpers";

beforeEach(() => stubEventSource());
afterEach(() => vi.unstubAllGlobals());

const assets = { "GET /api/assets": () => json([assetDetail()]) };
const running = job({ id: "job_run", status: "running", progress: 0.4, message: "shots 40%", started_at: "2026-10-05T08:00:05Z" });
const failed = job({
  id: "job_bad",
  stage: "creation.script",
  status: "failed",
  error: "LLMAuthError: bad key",
  attempt: 2,
  created_at: "2026-10-05T07:00:00Z",
  started_at: "2026-10-05T07:00:01Z",
  finished_at: "2026-10-05T07:03:06Z",
});
const done = job({ id: "job_ok", stage: "output.render", status: "succeeded", progress: 1, created_at: "2026-10-05T06:00:00Z", started_at: "2026-10-05T06:00:01Z", finished_at: "2026-10-05T06:00:43Z" });

describe("job helpers", () => {
  it("labels stages and falls back to the raw name", () => {
    expect(stageLabel("creation.plan")).toBe("构建剪辑计划");
    expect(stageLabel("analysis.shots")).toBe("analysis.shots");
  });

  it("measures how long a finished job ran", () => {
    expect(runtime(done)).toBe("42 秒");
    expect(runtime(failed)).toBe("3 分 05 秒");
    expect(runtime(running)).toBeNull();
  });

  it("filters", () => {
    const all = [running, failed, done];
    expect(filterJobs(all, "all")).toHaveLength(3);
    expect(filterJobs(all, "active")).toEqual([running]);
    expect(filterJobs(all, "failed")).toEqual([failed]);
  });
});

describe("job center", () => {
  it("lists jobs with state, progress, asset, attempts, runtime and error", async () => {
    stubApi({ ...assets, "GET /api/jobs": () => json([running, failed, done]) });
    renderWithProviders(<App />, "/jobs");
    const run = await screen.findByRole("listitem", { name: /分析影片/ });
    expect(run).toHaveTextContent("运行中");
    expect(run).toHaveTextContent("shots 40%");
    expect(within(run).getByRole("progressbar", { name: "进度" })).toHaveValue(0.4);
    await waitFor(() => expect(run).toHaveTextContent("Sintel")); // asset title, once the assets load

    const bad = screen.getByRole("listitem", { name: /生成文案/ });
    expect(bad).toHaveTextContent("失败");
    expect(bad).toHaveTextContent("第 2 次尝试");
    expect(bad).toHaveTextContent("用时 3 分 05 秒");
    expect(within(bad).getByRole("alert")).toHaveTextContent("LLMAuthError: bad key");

    const ok = screen.getByRole("listitem", { name: /渲染成片/ });
    expect(ok).toHaveTextContent("已完成");
    expect(within(ok).queryByRole("progressbar")).not.toBeInTheDocument();
  });

  it("filters by active and failed", async () => {
    stubApi({ ...assets, "GET /api/jobs": () => json([running, failed, done]) });
    renderWithProviders(<App />, "/jobs");
    await screen.findByRole("listitem", { name: /分析影片/ });
    await userEvent.click(screen.getByRole("button", { name: "进行中" }));
    expect(screen.getAllByRole("listitem")).toHaveLength(1);
    await userEvent.click(screen.getByRole("button", { name: "失败" }));
    expect(screen.getByRole("listitem")).toHaveTextContent("生成文案");
    await userEvent.click(screen.getByRole("button", { name: "全部" }));
    expect(screen.getAllByRole("listitem")).toHaveLength(3);
  });

  it("says so when nothing matches", async () => {
    stubApi({ ...assets, "GET /api/jobs": () => json([done]) });
    renderWithProviders(<App />, "/jobs");
    await screen.findByRole("listitem");
    await userEvent.click(screen.getByRole("button", { name: "失败" }));
    expect(screen.getByText("没有符合条件的作业。")).toBeInTheDocument();
  });

  it("offers cancel only on active jobs and retry only on failed ones", async () => {
    stubApi({ ...assets, "GET /api/jobs": () => json([running, failed, done]) });
    renderWithProviders(<App />, "/jobs");
    const run = await screen.findByRole("listitem", { name: /分析影片/ });
    const bad = screen.getByRole("listitem", { name: /生成文案/ });
    const ok = screen.getByRole("listitem", { name: /渲染成片/ });
    expect(within(run).getByRole("button", { name: "取消" })).toBeInTheDocument();
    expect(within(run).queryByRole("button", { name: "重试" })).not.toBeInTheDocument();
    expect(within(bad).getByRole("button", { name: "重试" })).toBeInTheDocument();
    expect(within(bad).queryByRole("button", { name: "取消" })).not.toBeInTheDocument();
    expect(within(ok).queryByRole("button", { name: "取消" })).not.toBeInTheDocument();
    expect(within(ok).queryByRole("button", { name: "重试" })).not.toBeInTheDocument();
  });

  it("cancels a queued job and shows the result at once", async () => {
    const queued = job({ id: "job_q", status: "queued" });
    const cancel = vi.fn(() => json({ ...queued, status: "canceled" }));
    stubApi({ ...assets, "GET /api/jobs": () => json([queued]), "POST /api/jobs/job_q:cancel": cancel });
    renderWithProviders(<App />, "/jobs");
    await userEvent.click(await screen.findByRole("button", { name: "取消" }));
    await waitFor(() => expect(screen.getByRole("listitem")).toHaveTextContent("已取消"));
    expect(cancel).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("button", { name: "取消" })).not.toBeInTheDocument();
  });

  it("shows a running job as canceling until it actually stops", async () => {
    const cancel = vi.fn(() => json(running)); // still running: the stage stops at its next checkpoint
    stubApi({ ...assets, "GET /api/jobs": () => json([running]), "POST /api/jobs/job_run:cancel": cancel });
    renderWithProviders(<App />, "/jobs");
    await userEvent.click(await screen.findByRole("button", { name: "取消" }));
    const row = screen.getByRole("listitem");
    await waitFor(() => expect(row).toHaveTextContent("取消中…"));
    expect(within(row).getByRole("button", { name: "取消" })).toBeDisabled();
    // the live feed then reports the end
    FakeEventSource.instances[0]?.emit("job", { ...running, status: "canceled" });
    await waitFor(() => expect(row).toHaveTextContent("已取消"));
  });

  it("retries a failed job", async () => {
    const retry = vi.fn(() => json({ ...failed, status: "queued", error: null, attempt: 0 }));
    stubApi({ ...assets, "GET /api/jobs": () => json([failed]), "POST /api/jobs/job_bad:retry": retry });
    renderWithProviders(<App />, "/jobs");
    await userEvent.click(await screen.findByRole("button", { name: "重试" }));
    await waitFor(() => expect(screen.getByRole("listitem")).toHaveTextContent("排队中"));
    expect(retry).toHaveBeenCalledTimes(1);
  });

  it("shows why a control action was refused", async () => {
    stubApi({
      ...assets,
      "GET /api/jobs": () => json([failed]),
      "POST /api/jobs/job_bad:retry": () => json({ error: { code: "conflict", message: "job job_bad is running; only failed jobs can be retried" } }, 409),
    });
    renderWithProviders(<App />, "/jobs");
    await userEvent.click(await screen.findByRole("button", { name: "重试" }));
    await waitFor(() => expect(screen.getAllByRole("alert").some((a) => a.textContent?.includes("only failed jobs"))).toBe(true));
  });

  it("follows live updates without a refetch", async () => {
    stubApi({ ...assets, "GET /api/jobs": () => json([job({ id: "job_run", status: "running", progress: 0.1 })]) });
    renderWithProviders(<App />, "/jobs");
    const row = await screen.findByRole("listitem");
    expect(within(row).getByRole("progressbar")).toHaveValue(0.1);
    FakeEventSource.instances[0]?.emit("job", { ...running, progress: 0.8, message: "story 80%" });
    await waitFor(() => expect(within(row).getByRole("progressbar")).toHaveValue(0.8));
    expect(row).toHaveTextContent("story 80%");
    FakeEventSource.instances[0]?.emit("job", job({ id: "job_new", stage: "output.render", created_at: "2026-10-05T12:00:00Z" }));
    expect(await screen.findByRole("listitem", { name: /渲染成片/ })).toBeInTheDocument();
  });
});

describe("job log", () => {
  it("opens and closes the log of a job", async () => {
    const log = vi.fn(() => new Response("start analysis.story\nsucceeded\n", { headers: { "content-type": "text/plain" } }));
    stubApi({ ...assets, "GET /api/jobs": () => json([done]), "GET /api/jobs/job_ok/log": log });
    renderWithProviders(<App />, "/jobs");
    const toggle = await screen.findByRole("button", { name: "日志" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(toggle);
    const pre = await screen.findByLabelText("作业日志");
    expect(pre).toHaveTextContent("start analysis.story");
    expect(pre).toHaveTextContent("succeeded");
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    await userEvent.click(toggle);
    expect(screen.queryByLabelText("作业日志")).not.toBeInTheDocument();
  });

  it("says when there is no log yet", async () => {
    stubApi({ ...assets, "GET /api/jobs": () => json([job()]), "GET /api/jobs/job_1/log": () => new Response("", { headers: { "content-type": "text/plain" } }) });
    renderWithProviders(<App />, "/jobs");
    await userEvent.click(await screen.findByRole("button", { name: "日志" }));
    expect(await screen.findByLabelText("作业日志")).toHaveTextContent("（还没有日志）");
  });

  it("reports a log that cannot be read", async () => {
    stubApi({
      ...assets,
      "GET /api/jobs": () => json([done]),
      "GET /api/jobs/job_ok/log": () => json({ error: { code: "not_found", message: "unknown job job_ok" } }, 404),
    });
    renderWithProviders(<App />, "/jobs");
    await userEvent.click(await screen.findByRole("button", { name: "日志" }));
    expect(await screen.findByText(/读取日志失败：unknown job job_ok/)).toBeInTheDocument();
  });

  it("keeps refreshing the log of a running job", async () => {
    let calls = 0;
    const log = () => new Response(`line ${++calls}\n`, { headers: { "content-type": "text/plain" } });
    stubApi({ ...assets, "GET /api/jobs": () => json([running]), "GET /api/jobs/job_run/log": log });
    renderWithProviders(<App />, "/jobs");
    await userEvent.click(await screen.findByRole("button", { name: "日志" }));
    await screen.findByLabelText("作业日志");
    await waitFor(() => expect(calls).toBeGreaterThanOrEqual(2), { timeout: 4000 });
  });
});

describe("unwrap with text responses", () => {
  it("reads the error shape out of a text body", async () => {
    stubApi({ "GET /api/jobs/job_x/log": () => json({ error: { code: "not_found", message: "unknown job job_x" } }, 404) });
    const result = await api.GET("/api/jobs/{job_id}/log", { params: { path: { job_id: "job_x" } }, parseAs: "text" });
    expect(() => unwrap(result)).toThrow(expect.objectContaining({ status: 404, code: "not_found", message: "unknown job job_x" }));
  });
});
