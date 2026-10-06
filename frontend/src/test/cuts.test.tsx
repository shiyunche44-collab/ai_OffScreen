import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { cutsFromShots, frameAt, neighbours, sameCuts, timeOfFrame, withCut, withoutCut } from "../lib/cuts";
import { assetDetail, json, renderWithProviders, stubApi, stubEventSource } from "./helpers";

beforeEach(() => stubEventSource());
afterEach(() => vi.unstubAllGlobals());

describe("cut arithmetic", () => {
  it("maps seconds to frames and back through the middle of a frame", () => {
    expect(frameAt(0, 24)).toBe(0);
    expect(frameAt(1, 24)).toBe(24);
    expect(frameAt(timeOfFrame(7, 23.976), 23.976)).toBe(7);
    for (let f = 0; f < 500; f += 13) expect(frameAt(timeOfFrame(f, 24000 / 1001), 24000 / 1001)).toBe(f);
    expect(frameAt(-1, 24)).toBe(0);
  });

  it("keeps cuts ascending, unique and never at frame 0", () => {
    expect(withCut([10, 30], 20)).toEqual([10, 20, 30]);
    expect(withCut([10, 30], 10)).toEqual([10, 30]);
    expect(withCut([10], 0)).toEqual([10]);
    expect(withoutCut([10, 20, 30], 20)).toEqual([10, 30]);
  });

  it("finds the cuts around a frame", () => {
    expect(neighbours([10, 20, 30], 20)).toEqual({ prev: 10, next: 30 });
    expect(neighbours([10, 20, 30], 5)).toEqual({ prev: null, next: 10 });
    expect(neighbours([10, 20, 30], 99)).toEqual({ prev: 30, next: null });
    expect(neighbours([], 3)).toEqual({ prev: null, next: null });
  });

  it("turns shot starts into cut frames", () => {
    expect(cutsFromShots([0, 8000, 16000], 24)).toEqual([192, 384]);
    expect(cutsFromShots([8000, 8001], 24)).toEqual([192]);
    expect(sameCuts([1, 2], [1, 2])).toBe(true);
    expect(sameCuts([1, 2], [1, 3])).toBe(false);
  });
});

const detail = () => ({
  ...assetDetail({ cached: [true, true, true, true] }),
  asset: { ...assetDetail().asset, video: { width: 1280, height: 544, fps: { num: 24, den: 1 }, codec: "h264" } },
});

const shotsView = {
  asset_id: "ast_1",
  video: "artifacts/p/proxy_540p.mp4",
  tile_width: 160,
  tile_height: 90,
  columns: 10,
  rows: 10,
  sheets: [],
  shots: [0, 1, 2].map((n) => ({
    id: `sh_${n}`,
    start_ms: n * 8000,
    end_ms: (n + 1) * 8000,
    keyframes: [],
    caption: null,
    sprite: null,
  })),
};

const view = (cuts: number[], marked = true) => ({ asset_id: "ast_1", fps_num: 24, fps_den: 1, cuts, marked });
const evaluation = (over: Record<string, unknown> = {}) => ({
  asset_id: "ast_1",
  source: "shots",
  tolerance: 2,
  marked: 3,
  detected: 2,
  true_positives: 2,
  precision: 1,
  recall: 0.6667,
  f1: 0.8,
  false_positives: [],
  false_negatives: [300],
  ...over,
});

function base(extra: Record<string, (r: Request) => Response | Promise<Response>> = {}) {
  return {
    "GET /api/jobs": () => json([]),
    "GET /api/assets/ast_1": () => json(detail()),
    "GET /api/assets/ast_1/index/shots": () => json(shotsView),
    "GET /api/assets/ast_1/annotations/cuts": () => json(view([], false)),
    ...extra,
  };
}

function setTime(video: HTMLVideoElement, seconds: number) {
  Object.defineProperty(video, "currentTime", { value: seconds, writable: true, configurable: true });
  fireEvent.timeUpdate(video);
}

describe("cuts page", () => {
  it("is reached from the analysis page", async () => {
    stubApi({
      ...base(),
      "GET /api/assets/ast_1/index/transcript": () => json({ asset_id: "ast_1", language: "en", source: "x", lines: [] }),
    });
    renderWithProviders(<App />, "/library/ast_1");
    await userEvent.click(await screen.findByRole("link", { name: "标注切点" }));
    expect(await screen.findByRole("heading", { name: /切点标注：Sintel/ })).toBeInTheDocument();
  });

  it("marks the frame the video is on and shows the cut", async () => {
    stubApi(base());
    renderWithProviders(<App />, "/library/ast_1/cuts");
    const video = (await screen.findByLabelText("代理视频")) as HTMLVideoElement;
    expect(screen.getByRole("button", { name: "标记为切点" })).toBeDisabled(); // frame 0 is no cut

    setTime(video, 8.02); // frame 192
    await waitFor(() => expect(screen.getByLabelText("当前帧")).toHaveTextContent("帧 192"));
    await userEvent.click(screen.getByRole("button", { name: "标记为切点" }));
    expect(screen.getByRole("heading", { name: "已标记 1 个切点" })).toBeInTheDocument();
    expect(within(screen.getByRole("list", { name: "切点" })).getByText("帧 192")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "已标记" })).toBeDisabled();
    expect(screen.getByText("有未保存的修改")).toBeInTheDocument();
  });

  it("steps frame by frame with the buttons and the keys", async () => {
    stubApi(base());
    renderWithProviders(<App />, "/library/ast_1/cuts");
    const video = (await screen.findByLabelText("代理视频")) as HTMLVideoElement;
    Object.defineProperty(video, "currentTime", { value: 0, writable: true, configurable: true });

    await userEvent.click(screen.getByRole("button", { name: "下一帧" }));
    await userEvent.click(screen.getByRole("button", { name: "下一帧" }));
    expect(screen.getByLabelText("当前帧")).toHaveTextContent("帧 2");
    expect(video.currentTime).toBeCloseTo(2.5 / 24, 6); // the middle of frame 2
    await userEvent.keyboard(".");
    expect(screen.getByLabelText("当前帧")).toHaveTextContent("帧 3");
    await userEvent.keyboard(",,,,,"); // never below frame 0
    expect(screen.getByLabelText("当前帧")).toHaveTextContent("帧 0");
    await userEvent.keyboard(".....m");
    expect(screen.getByRole("heading", { name: "已标记 1 个切点" })).toBeInTheDocument();
    expect(within(screen.getByRole("list", { name: "切点" })).getByText("帧 5")).toBeInTheDocument();
  });

  it("starts from the detected cuts, lets some be removed, and saves", async () => {
    const put = vi.fn(async (req: Request) => {
      expect(await req.json()).toEqual({ cuts: [384] });
      return json(view([384]));
    });
    stubApi(
      base({
        "PUT /api/assets/ast_1/annotations/cuts": put,
        "GET /api/assets/ast_1/annotations/cuts/evaluation": () => json(evaluation()),
      }),
    );
    renderWithProviders(<App />, "/library/ast_1/cuts");
    await userEvent.click(await screen.findByRole("button", { name: "载入检测结果作为起点" }));
    const list = screen.getByRole("list", { name: "切点" });
    expect(within(list).getAllByRole("listitem")).toHaveLength(2); // 192 and 384
    await userEvent.click(screen.getByRole("button", { name: "删除切点 192" }));
    await userEvent.click(screen.getByRole("button", { name: "保存" }));
    await waitFor(() => expect(put).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.queryByText("有未保存的修改")).not.toBeInTheDocument());
    expect(screen.getByRole("heading", { name: "已标记 1 个切点" })).toBeInTheDocument();
  });

  it("discards unsaved changes", async () => {
    stubApi(base({ "GET /api/assets/ast_1/annotations/cuts": () => json(view([192])) }));
    renderWithProviders(<App />, "/library/ast_1/cuts");
    await screen.findByRole("heading", { name: "已标记 1 个切点" });
    await userEvent.click(screen.getByRole("button", { name: "删除切点 192" }));
    expect(screen.getByRole("heading", { name: "已标记 0 个切点" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "放弃修改" }));
    expect(screen.getByRole("heading", { name: "已标记 1 个切点" })).toBeInTheDocument();
  });

  it("jumps to a cut and to its neighbours", async () => {
    stubApi(base({ "GET /api/assets/ast_1/annotations/cuts": () => json(view([100, 200, 300])) }));
    renderWithProviders(<App />, "/library/ast_1/cuts");
    const video = (await screen.findByLabelText("代理视频")) as HTMLVideoElement;
    Object.defineProperty(video, "currentTime", { value: 0, writable: true, configurable: true });
    await userEvent.click(await screen.findByRole("button", { name: "帧 200" }));
    expect(screen.getByLabelText("当前帧")).toHaveTextContent("帧 200");
    await userEvent.click(screen.getByRole("button", { name: "上一个切点" }));
    expect(screen.getByLabelText("当前帧")).toHaveTextContent("帧 100");
    expect(screen.getByRole("button", { name: "上一个切点" })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "下一个切点" }));
    await userEvent.click(screen.getByRole("button", { name: "下一个切点" }));
    expect(screen.getByLabelText("当前帧")).toHaveTextContent("帧 300");
  });

  it("shows how the detector scores against the saved marks, with a tolerance", async () => {
    const tolerances: string[] = [];
    stubApi(
      base({
        "GET /api/assets/ast_1/annotations/cuts": () => json(view([192, 384, 300])),
        "GET /api/assets/ast_1/annotations/cuts/evaluation": (req) => {
          const t = new URL(req.url).searchParams.get("tolerance") ?? "";
          tolerances.push(t);
          return json(evaluation({ tolerance: Number(t), f1: t === "2" ? 0.8 : 1, false_negatives: t === "2" ? [300] : [] }));
        },
      }),
    );
    renderWithProviders(<App />, "/library/ast_1/cuts");
    const box = await screen.findByLabelText("评估");
    expect(await within(box).findByText(/F1/)).toHaveTextContent("准确率 1.000 · 召回率 0.667 · F1 0.800");
    expect(box).toHaveTextContent("标记 3 · 检测 2 · 命中 2");
    expect(box).toHaveTextContent("漏掉的（帧）：300");
    fireEvent.change(within(box).getByRole("spinbutton"), { target: { value: "5" } });
    await waitFor(() => expect(box).toHaveTextContent("F1 1.000"));
    expect(tolerances).toEqual(["2", "5"]);
  });

  it("explains that scores need saved marks", async () => {
    stubApi(base());
    renderWithProviders(<App />, "/library/ast_1/cuts");
    expect(await screen.findByText(/保存标记后/)).toBeInTheDocument();
  });

  it("shows a failed save", async () => {
    stubApi(
      base({
        "PUT /api/assets/ast_1/annotations/cuts": () =>
          json({ error: { code: "invalid_input", message: "frame 9999 is past the end of the film" } }, 422),
      }),
    );
    renderWithProviders(<App />, "/library/ast_1/cuts");
    await userEvent.click(await screen.findByRole("button", { name: "载入检测结果作为起点" }));
    await userEvent.click(screen.getByRole("button", { name: "保存" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("past the end of the film");
  });
});
