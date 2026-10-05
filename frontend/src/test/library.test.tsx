import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { assetDetail, job, json, renderWithProviders, stubApi, stubEventSource } from "./helpers";

beforeEach(() => stubEventSource());
afterEach(() => vi.unstubAllGlobals());

const noJobs = { "GET /api/jobs": () => json([]) };

describe("library: asset list", () => {
  it("shows what each asset is and where its analysis stands", async () => {
    stubApi({
      ...noJobs,
      "GET /api/assets": () =>
        json([
          assetDetail({ id: "ast_1", title: "Sintel", cached: [true, true, true, true] }),
          assetDetail({ id: "ast_2", title: "Tears of Steel", subtitles: false }),
        ]),
    });
    renderWithProviders(<App />, "/library");
    const sintel = await screen.findByRole("listitem", { name: "Sintel" });
    expect(sintel).toHaveTextContent("14:48 · 1280×544 · 有外挂字幕");
    expect(sintel).toHaveTextContent("已分析");
    expect(within(sintel).queryByRole("button", { name: "分析" })).not.toBeInTheDocument();
    const tears = screen.getByRole("listitem", { name: "Tears of Steel" });
    expect(tears).toHaveTextContent("未分析");
    expect(tears).toHaveTextContent("无外挂字幕");
    expect(within(tears).getByRole("button", { name: "分析" })).toBeEnabled();
  });

  it("shows analysis progress from the live job list and blocks a second analyze", async () => {
    stubApi({
      "GET /api/assets": () => json([assetDetail()]),
      "GET /api/jobs": () =>
        json([job({ stage: "analysis.story", status: "running", progress: 0.4, scope: { asset_id: "ast_1" } })]),
    });
    renderWithProviders(<App />, "/library");
    const row = await screen.findByRole("listitem", { name: "Sintel" });
    await waitFor(() => expect(row).toHaveTextContent("分析中 40%"));
    expect(within(row).getByRole("progressbar", { name: "分析进度" })).toHaveValue(0.4);
    expect(within(row).getByRole("button", { name: "分析" })).toBeDisabled();
  });

  it("shows a failed analysis and offers to retry it", async () => {
    stubApi({
      "GET /api/assets": () => json([assetDetail()]),
      "GET /api/jobs": () =>
        json([job({ stage: "analysis.story", status: "failed", error: "LLMAuthError: bad key", scope: { asset_id: "ast_1" } })]),
    });
    renderWithProviders(<App />, "/library");
    expect(await screen.findByText(/分析失败：LLMAuthError: bad key/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "重新分析" })).toBeEnabled();
  });

  it("queues the analysis when asked", async () => {
    const analyze = vi.fn(() => json(job({ status: "queued" }), 202));
    stubApi({ ...noJobs, "GET /api/assets": () => json([assetDetail()]), "POST /api/assets/ast_1/analyze": analyze });
    renderWithProviders(<App />, "/library");
    await userEvent.click(await screen.findByRole("button", { name: "分析" }));
    await waitFor(() => expect(analyze).toHaveBeenCalledTimes(1));
  });

  it("creates a project and opens it", async () => {
    const create = vi.fn(async (req: Request) => {
      expect(await req.json()).toEqual({ asset_id: "ast_1" });
      return json({ id: "prj_9", asset_id: "ast_1", name: "Sintel", options: {}, created_at: "2026-10-05T08:00:00Z" }, 201);
    });
    stubApi({
      ...noJobs,
      "GET /api/assets": () => json([assetDetail()]),
      "POST /api/projects": create,
      "GET /api/projects/prj_9": () =>
        json({
          project: { id: "prj_9", asset_id: "ast_1", name: "Sintel 项目", options: {}, created_at: "2026-10-05T08:00:00Z" },
          stages: [],
          video: null,
        }),
    });
    renderWithProviders(<App />, "/library");
    await userEvent.click(await screen.findByRole("button", { name: "新建项目" }));
    expect(await screen.findByRole("heading", { name: "Sintel 项目" })).toBeInTheDocument();
  });

  it("explains an empty library", async () => {
    stubApi({ ...noJobs, "GET /api/assets": () => json([]) });
    renderWithProviders(<App />, "/library");
    expect(await screen.findByText(/还没有素材/)).toBeInTheDocument();
  });
});

describe("library: importing", () => {
  it("imports by path, then clears the field and refreshes the list", async () => {
    let imported = false;
    const post = vi.fn(async (req: Request) => {
      expect(await req.json()).toEqual({ path: "/movies/Sintel.mkv" });
      imported = true;
      return json(assetDetail().asset, 201);
    });
    stubApi({
      ...noJobs,
      "GET /api/assets": () => json(imported ? [assetDetail()] : []),
      "POST /api/assets": post,
    });
    renderWithProviders(<App />, "/library");
    const input = await screen.findByRole("textbox", { name: "电影文件路径" });
    const submit = screen.getByRole("button", { name: "导入" });
    expect(submit).toBeDisabled(); // nothing typed yet
    await userEvent.type(input, "  /movies/Sintel.mkv ");
    await userEvent.click(submit);
    expect(await screen.findByText("已导入：Sintel")).toBeInTheDocument();
    expect(input).toHaveValue("");
    expect(await screen.findByRole("listitem", { name: "Sintel" })).toBeInTheDocument();
  });

  it("shows why an import was refused and keeps what was typed", async () => {
    stubApi({
      ...noJobs,
      "GET /api/assets": () => json([]),
      "POST /api/assets": () =>
        json({ error: { code: "invalid_input", message: "the file is outside the configured media roots" } }, 422),
    });
    renderWithProviders(<App />, "/library");
    const input = await screen.findByRole("textbox", { name: "电影文件路径" });
    await userEvent.type(input, "/etc/passwd");
    await userEvent.click(screen.getByRole("button", { name: "导入" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("outside the configured media roots");
    expect(input).toHaveValue("/etc/passwd");
  });
});

describe("library: browsing the media roots", () => {
  const listing = (path: string | null, parent: string | null, entries: object[]) => json({ path, parent, entries, truncated: false });

  it("walks folders and fills the path from a picked file", async () => {
    stubApi({
      ...noJobs,
      "GET /api/assets": () => json([]),
      "GET /api/assets/browse": (req) => {
        const path = new URL(req.url).searchParams.get("path");
        if (path === null) return listing(null, null, [{ name: "movies", path: "/media/movies", kind: "dir" }]);
        if (path === "/media/movies")
          return listing(path, null, [
            { name: "Extras", path: "/media/movies/Extras", kind: "dir" },
            { name: "Sintel.mkv", path: "/media/movies/Sintel.mkv", kind: "file", size: 681_285_280 },
            { name: "Seen.mkv", path: "/media/movies/Seen.mkv", kind: "file", size: 1024, asset_id: "ast_7" },
          ]);
        return listing(path, "/media/movies", []);
      },
    });
    renderWithProviders(<App />, "/library");
    await userEvent.click(await screen.findByRole("button", { name: "浏览…" }));
    const browser = await screen.findByLabelText("媒体目录");
    expect(within(browser).getByText("媒体根目录")).toBeInTheDocument();
    expect(within(browser).getByRole("button", { name: "上一级" })).toBeDisabled();

    await userEvent.click(await within(browser).findByRole("button", { name: /movies/ }));
    const sintel = await within(browser).findByRole("button", { name: /Sintel\.mkv/ });
    expect(sintel).toHaveTextContent("650 MB");
    expect(within(browser).getByRole("button", { name: /Seen\.mkv/ })).toBeDisabled(); // already imported
    expect(within(browser).getByRole("button", { name: /Seen\.mkv/ })).toHaveTextContent("已导入");

    await userEvent.click(within(browser).getByRole("button", { name: /Extras/ }));
    expect(await within(browser).findByText("这里没有文件夹或视频。")).toBeInTheDocument();
    await userEvent.click(within(browser).getByRole("button", { name: "上一级" }));
    await userEvent.click(await within(browser).findByRole("button", { name: /Sintel\.mkv/ }));
    expect(screen.getByRole("textbox", { name: "电影文件路径" })).toHaveValue("/media/movies/Sintel.mkv");
  });

  it("says what to configure when there are no media roots", async () => {
    stubApi({
      ...noJobs,
      "GET /api/assets": () => json([]),
      "GET /api/assets/browse": () =>
        json({ error: { code: "invalid_input", message: "no media_roots configured; add the folder holding your movies" } }, 422),
    });
    renderWithProviders(<App />, "/library");
    await userEvent.click(await screen.findByRole("button", { name: "浏览…" }));
    const alerts = await screen.findAllByRole("alert");
    expect(alerts.some((a) => a.textContent?.includes("media_roots"))).toBe(true);
  });
});
