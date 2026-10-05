import { screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { FakeEventSource, assetDetail, job, json, renderWithProviders, stubApi, stubEventSource } from "./helpers";

beforeEach(() => stubEventSource());
afterEach(() => vi.unstubAllGlobals());

describe("routing and layout", () => {
  it("redirects / to the library and shows the navigation", async () => {
    stubApi({ "GET /api/assets": () => json([assetDetail()]) });
    renderWithProviders(<App />, "/");
    expect(await screen.findByRole("heading", { name: "素材库" })).toBeInTheDocument();
    expect(await screen.findByText("Sintel")).toBeInTheDocument();
    const nav = screen.getByRole("navigation", { name: "主导航" });
    expect(nav).toHaveTextContent("素材库");
    expect(nav).toHaveTextContent("作业中心");
  });

  it("shows the job center with live progress", async () => {
    stubApi({ "GET /api/jobs": () => json([job({ status: "running", progress: 0.4, message: "shots 40%" })]) });
    renderWithProviders(<App />, "/jobs");
    expect(await screen.findByText("分析影片")).toBeInTheDocument();
    expect(screen.getByText("shots 40%")).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "进度" })).toHaveValue(0.4);
  });

  it("shows a project with the stages that are built", async () => {
    stubApi({
      "GET /api/projects/prj_1": () =>
        json({
          project: { id: "prj_1", asset_id: "ast_1", name: "三分钟版", options: {}, created_at: "2026-10-05T08:00:00Z" },
          stages: [
            { stage: "analysis.proxy", cached: true },
            { stage: "creation.script", cached: false },
          ],
          video: null,
        }),
    });
    renderWithProviders(<App />, "/projects/prj_1");
    expect(await screen.findByRole("heading", { name: "三分钟版" })).toBeInTheDocument();
    expect(screen.getByText(/analysis\.proxy/)).toHaveTextContent("✓");
    expect(screen.getByText(/creation\.script/)).toHaveTextContent("·");
  });

  it("explains an API failure instead of crashing", async () => {
    stubApi({
      "GET /api/projects/prj_x": () => json({ error: { code: "not_found", message: "unknown project prj_x" } }, 404),
    });
    renderWithProviders(<App />, "/projects/prj_x");
    expect(await screen.findByRole("alert")).toHaveTextContent("unknown project prj_x");
  });

  it("has a page for unknown addresses", () => {
    stubApi({});
    renderWithProviders(<App />, "/nowhere");
    expect(screen.getByRole("heading", { name: "页面不存在" })).toBeInTheDocument();
  });

  it("shows the live connection state", async () => {
    stubApi({ "GET /api/assets": () => json([]) });
    renderWithProviders(<App />, "/library");
    expect(screen.getByRole("status")).toHaveTextContent("连接中");
    FakeEventSource.instances[0]?.open();
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("实时"));
    FakeEventSource.instances[0]?.fail(true);
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("已断开"));
  });
});
