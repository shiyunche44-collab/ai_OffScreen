import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PlanEditorPage } from "../pages/PlanEditor";
import { job, json, renderWithProviders, stubApi } from "./helpers";

afterEach(() => vi.unstubAllGlobals());

const PROJECT = {
  project: { id: "prj_1", asset_id: "ast_1", name: "三分钟版", options: { minutes: 1 }, created_at: "2026-10-05T08:00:00Z" },
  stages: [{ stage: "creation.plan", cached: true }],
  video: null,
};

const SHOTS = {
  asset_id: "ast_1",
  video: "artifacts/analysis.proxy/abc/proxy.mp4",
  sheets: ["artifacts/a/sheet0.jpg"],
  columns: 10,
  rows: 10,
  tile_width: 160,
  tile_height: 90,
  shots: [
    { id: "sh_1", start_ms: 0, end_ms: 10_000, keyframes: [], sprite: { sheet: 0, col: 0, row: 0 } },
    { id: "sh_2", start_ms: 10_000, end_ms: 20_000, keyframes: [], sprite: { sheet: 0, col: 1, row: 0 } },
    { id: "sh_3", start_ms: 20_000, end_ms: 60_000, keyframes: [], sprite: { sheet: 0, col: 2, row: 0 } },
  ],
};

const LINES = {
  asset_id: "ast_1",
  language: "zh",
  source: "subtitle:external",
  lines: [
    { id: "ln_1", start_ms: 1_000, end_ms: 3_000, text: "我们不能留在这里。" },
    { id: "ln_2", start_ms: 40_000, end_ms: 42_000, text: "再见了，小龙。" },
  ],
};

const clip = (shot_id: string, src_in_ms: number, src_out_ms: number, locked = false) => ({
  asset_id: "ast_1",
  shot_id,
  src_in_ms,
  src_out_ms,
  speed: 1,
  locked,
  score: 0.8,
});

function narration(id: string, text: string, over: object = {}) {
  return {
    id,
    kind: "narration",
    text,
    text_hash: null,
    stale: false,
    voice: { voice_id: "v", speed: 1 },
    voice_pinned: false,
    line_refs: [],
    audio: { file: `tts/${id}.mp3`, duration_ms: 4_000, char_timings: [] },
    clips: [clip("sh_1", 3_000, 7_000)],
    source_audio: { mode: "duck", stem: "mix", gain_db: -20 },
    ...over,
  };
}

function plan(version = 1, segments: object[] = [narration("seg_01", "第一段解说"), narration("seg_02", "第二段解说", { clips: [clip("sh_2", 12_000, 14_000, true), clip("sh_3", 30_000, 32_000)] })]) {
  return {
    schema_version: 1,
    id: "pln_1",
    project_id: "prj_1",
    version,
    parent_version: version > 1 ? version - 1 : null,
    author: version > 1 ? "human" : "ai",
    script_ref: { id: "scr_1", version: 1 },
    segments,
    bgm: null,
    output_profile: "source",
  };
}

const CANDIDATES = {
  segment_id: "seg_01",
  plan_version: 1,
  total: 30,
  vector_search: true,
  candidates: [
    {
      shot_id: "sh_1",
      scene_id: "sc_001",
      start_ms: 0,
      end_ms: 10_000,
      score: 0.91,
      factors: { embedding: 0.9, caption: 0.8, character: 0, quality: 0.7, size: 0.8, reuse: 1, time_order: 1, exclusions: 1 },
      caption: "女孩抱着小龙",
      used_by: null,
      current: true,
    },
    {
      shot_id: "sh_3",
      scene_id: "sc_002",
      start_ms: 20_000,
      end_ms: 60_000,
      score: 0.62,
      factors: { embedding: 0.5, caption: 0.4, character: 0, quality: 0.7, size: 0.8, reuse: 0, time_order: 1, exclusions: 1 },
      caption: "雪山",
      used_by: "seg_02",
      current: false,
    },
  ],
};

type Handler = (r: Request) => Response | Promise<Response>;

function routes(over: Record<string, Handler> = {}): Record<string, Handler> {
  return {
    "GET /api/projects/prj_1": () => json(PROJECT),
    "GET /api/projects/prj_1/plan": () => json(plan()),
    "GET /api/assets/ast_1/index/shots": () => json(SHOTS),
    "GET /api/assets/ast_1/index/transcript": () => json(LINES),
    "GET /api/projects/prj_1/plan/segments/seg_01/candidates": () => json(CANDIDATES),
    "GET /api/jobs": () => json([]),
    ...over,
  };
}

function open() {
  return renderWithProviders(
    <Routes>
      <Route path="projects/:projectId/plan" element={<PlanEditorPage />} />
    </Routes>,
    "/projects/prj_1/plan",
  );
}

const rows = () => within(screen.getByRole("list", { name: "段落" })).getAllByRole("listitem", { name: /^第 \d+ 段$/ });

/** Records the body of every plan edit and answers with the plan the test says is next. */
function editRecorder(next: object = plan(2)) {
  const bodies: { base_version: number; ops: Record<string, unknown>[] }[] = [];
  const handler: Handler = async (r) => {
    bodies.push((await r.json()) as (typeof bodies)[number]);
    return json(next);
  };
  return { bodies, handler };
}

describe("plan editor", () => {
  beforeEach(() => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
  });

  it("lists the segments with their words, audio, clips and the plan's length", async () => {
    stubApi(routes());
    open();
    expect(await screen.findByRole("heading", { name: "剪辑计划" })).toBeInTheDocument();
    await waitFor(() => expect(rows()).toHaveLength(2));
    expect(screen.getByLabelText("计划概况")).toHaveTextContent("当前 v1（自动构建）· 共 2 段 · 0:08");
    expect(within(rows()[0]!).getByText("第一段解说")).toBeInTheDocument();
    expect(within(rows()[0]!).getByLabelText("seg_01 配音")).toHaveAttribute("src", "/api/files/projects/prj_1/docs/plan/tts/seg_01.mp3");
    expect(within(rows()[1]!).getAllByRole("listitem", { name: /^镜头 \d+$/ })).toHaveLength(2);
    expect(within(rows()[1]!).getByRole("button", { name: "解锁镜头 1" })).toHaveAttribute("aria-pressed", "true");
    expect(within(rows()[1]!).getByRole("button", { name: "锁定镜头 2" })).toHaveAttribute("aria-pressed", "false");
  });

  it("says so when there is no plan yet", async () => {
    stubApi(routes({ "GET /api/projects/prj_1/plan": () => json({ error: { code: "not_found", message: "no plan" } }, 404) }));
    open();
    expect(await screen.findByText(/还没有剪辑计划/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "构建计划" })).toBeEnabled();
  });

  it("swaps a clip for a candidate, on the version it is looking at", async () => {
    const edits = editRecorder();
    stubApi(routes({ "POST /api/projects/prj_1/plan:edit": edits.handler }));
    const user = userEvent.setup();
    open();
    await waitFor(() => expect(rows()).toHaveLength(2));
    await user.click(within(rows()[0]!).getByRole("button", { name: "替换镜头 1" }));

    const drawer = await screen.findByRole("complementary", { name: "候选镜头" });
    const list = await within(drawer).findByRole("list", { name: "候选列表" });
    const cards = within(list).getAllByRole("listitem");
    expect(cards[0]).toHaveTextContent("91 分");
    expect(cards[0]).toHaveTextContent("正在使用");
    expect(within(cards[0]!).getByRole("button", { name: "用候选 1" })).toBeDisabled();
    expect(cards[1]).toHaveTextContent("已用于 seg_02");

    await user.click(within(cards[1]!).getByRole("button", { name: "用候选 2" }));
    await waitFor(() => expect(edits.bodies).toHaveLength(1));
    expect(edits.bodies[0]).toEqual({
      base_version: 1,
      ops: [{ op: "swap_clip", segment_id: "seg_01", index: 0, to: { shot_id: "sh_3", speed: 1 } }],
    });
    await waitFor(() => expect(screen.queryByRole("complementary", { name: "候选镜头" })).toBeNull());
  });

  it("takes any stretch of the film for a clip, and checks it first", async () => {
    const edits = editRecorder();
    stubApi(routes({ "POST /api/projects/prj_1/plan:edit": edits.handler }));
    const user = userEvent.setup();
    open();
    await waitFor(() => expect(rows()).toHaveLength(2));
    await user.click(within(rows()[0]!).getByRole("button", { name: "替换镜头 1" }));
    const drawer = await screen.findByRole("complementary", { name: "候选镜头" });
    await user.click(within(drawer).getByText("用影片的任意一段"));
    const use = within(drawer).getByRole("button", { name: "用这一段" });
    expect(use).toBeDisabled();
    await user.type(within(drawer).getByLabelText("区间入点（秒）"), "50");
    await user.type(within(drawer).getByLabelText("区间出点（秒）"), "40");
    expect(use).toBeDisabled();
    expect(within(drawer).getByText("出点要在入点之后")).toBeInTheDocument();
    await user.clear(within(drawer).getByLabelText("区间出点（秒）"));
    await user.type(within(drawer).getByLabelText("区间出点（秒）"), "52.5");
    await user.click(use);
    await waitFor(() => expect(edits.bodies).toHaveLength(1));
    expect(edits.bodies[0]!.ops).toEqual([{ op: "swap_clip", segment_id: "seg_01", index: 0, to: { src_in_ms: 50_000, src_out_ms: 52_500, speed: 1 } }]);
  });

  it("locks, moves and deletes", async () => {
    const edits = editRecorder();
    stubApi(routes({ "POST /api/projects/prj_1/plan:edit": edits.handler }));
    const user = userEvent.setup();
    open();
    await waitFor(() => expect(rows()).toHaveLength(2));

    await user.click(within(rows()[0]!).getByRole("button", { name: "锁定镜头 1" }));
    await user.click(within(rows()[1]!).getByRole("button", { name: "上移 seg_02" }));
    await user.click(within(rows()[1]!).getByRole("button", { name: "移除镜头 2" }));
    await user.click(within(rows()[0]!).getByRole("button", { name: "删除 seg_01" }));
    await waitFor(() => expect(edits.bodies).toHaveLength(4));
    expect(edits.bodies.map((b) => b.ops[0])).toEqual([
      { op: "set_locked", segment_id: "seg_01", index: 0, locked: true },
      { op: "move_segment", segment_id: "seg_02", to_index: 0 },
      { op: "remove_clip", segment_id: "seg_02", index: 1 },
      { op: "delete_segment", segment_id: "seg_01" },
    ]);
    // the ends of the list cannot move further
    expect(within(rows()[0]!).getByRole("button", { name: "上移 seg_01" })).toBeDisabled();
    expect(within(rows()[1]!).getByRole("button", { name: "下移 seg_02" })).toBeDisabled();
  });

  it("trims a clip on the proxy film and sends the new interval", async () => {
    const edits = editRecorder();
    stubApi(routes({ "POST /api/projects/prj_1/plan:edit": edits.handler }));
    const user = userEvent.setup();
    open();
    await waitFor(() => expect(rows()).toHaveLength(2));
    await user.click(within(rows()[0]!).getByRole("button", { name: "裁剪镜头 1" }));

    const panel = await screen.findByRole("complementary", { name: "裁剪镜头" });
    expect(within(panel).getByLabelText("裁剪预览")).toHaveAttribute("src", "/api/files/artifacts/analysis.proxy/abc/proxy.mp4#t=3,7");
    const apply = within(panel).getByRole("button", { name: "应用（并锁定）" });
    expect(apply).toBeDisabled(); // nothing changed yet
    await user.clear(within(panel).getByLabelText("入点（秒）"));
    await user.type(within(panel).getByLabelText("入点（秒）"), "4.25");
    expect(within(panel).getByText(/播放 2.75 秒/)).toBeInTheDocument();
    await user.click(apply);
    await waitFor(() => expect(edits.bodies).toHaveLength(1));
    expect(edits.bodies[0]!.ops).toEqual([{ op: "trim_clip", segment_id: "seg_01", index: 0, src_in_ms: 4_250, src_out_ms: 7_000, speed: 1 }]);
  });

  it("refuses an out point past the end of the film", async () => {
    stubApi(routes());
    const user = userEvent.setup();
    open();
    await waitFor(() => expect(rows()).toHaveLength(2));
    await user.click(within(rows()[0]!).getByRole("button", { name: "裁剪镜头 1" }));
    const panel = await screen.findByRole("complementary", { name: "裁剪镜头" });
    await user.clear(within(panel).getByLabelText("出点（秒）"));
    await user.type(within(panel).getByLabelText("出点（秒）"), "61");
    expect(within(panel).getByRole("alert")).toHaveTextContent("出点超出影片（60 秒）");
    expect(within(panel).getByRole("button", { name: "应用（并锁定）" })).toBeDisabled();
  });

  it("inserts original sound after a segment from ticked transcript lines, in film order", async () => {
    const edits = editRecorder();
    stubApi(routes({ "POST /api/projects/prj_1/plan:edit": edits.handler }));
    const user = userEvent.setup();
    open();
    await waitFor(() => expect(rows()).toHaveLength(2));
    await user.click(within(rows()[0]!).getByRole("button", { name: "在 seg_01 之后插入原声" }));

    const panel = await screen.findByRole("complementary", { name: "插入原声" });
    const insert = within(panel).getByRole("button", { name: "插入" });
    expect(insert).toBeDisabled();
    await user.click(await within(panel).findByLabelText(/再见了，小龙/));
    await user.click(within(panel).getByLabelText(/我们不能留在这里/));
    await user.click(insert);
    await waitFor(() => expect(edits.bodies).toHaveLength(1));
    expect(edits.bodies[0]!.ops).toEqual([{ op: "insert_original", line_refs: ["ln_1", "ln_2"], after: "seg_01" }]);
  });

  it("changes a segment's voice and speed, and offers a rebuild for the stale ones", async () => {
    const next = plan(2, [narration("seg_01", "第一段解说", { stale: true, voice_pinned: true, voice: { voice_id: "mine", speed: 1.25 } })]);
    const edits = editRecorder(next);
    let built = false;
    stubApi(
      routes({
        "GET /api/projects/prj_1/plan": () => json(edits.bodies.length ? next : plan()),
        "POST /api/projects/prj_1/plan:edit": edits.handler,
        "POST /api/projects/prj_1/plan:build": () => {
          built = true;
          return json(job({ stage: "creation.plan", scope: { asset_id: "ast_1" } }), 202);
        },
      }),
    );
    const user = userEvent.setup();
    open();
    await waitFor(() => expect(rows()).toHaveLength(2));
    await user.click(within(rows()[0]!).getByText(/音色 v · 语速 1/));
    await user.clear(screen.getByLabelText("seg_01 音色"));
    await user.type(screen.getByLabelText("seg_01 音色"), "mine");
    await user.clear(screen.getByLabelText("seg_01 语速"));
    await user.type(screen.getByLabelText("seg_01 语速"), "3");
    expect(within(rows()[0]!).getByText("语速要在 0.5–2 之间")).toBeInTheDocument();
    await user.clear(screen.getByLabelText("seg_01 语速"));
    await user.type(screen.getByLabelText("seg_01 语速"), "1.25");
    await user.click(within(rows()[0]!).getByRole("button", { name: "应用" }));
    await waitFor(() => expect(edits.bodies).toHaveLength(1));
    expect(edits.bodies[0]!.ops).toEqual([{ op: "set_voice", segment_id: "seg_01", voice_id: "mine", speed: 1.25 }]);

    // the next version has the segment waiting for new audio
    expect(await screen.findByRole("status")).toHaveTextContent("有 1 段改了音色或语速");
    expect(within(rows()[0]!).getByText("待重新配音")).toBeInTheDocument();
    expect(within(rows()[0]!).getByRole("button", { name: "预览 seg_01" })).toBeDisabled();
    expect(within(rows()[0]!).getByRole("button", { name: "替换镜头 1" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "重建计划" }));
    await waitFor(() => expect(built).toBe(true));
  });

  it("previews one segment", async () => {
    let asked: URL | undefined;
    stubApi(
      routes({
        "POST /api/projects/prj_1/plan/segments/seg_01:preview": (r) => {
          asked = new URL(r.url);
          return json({ file: "previews/abc.mp4", cached: false, segment_hash: "h", duration_ms: 4_000, plan_version: 1 });
        },
      }),
    );
    const user = userEvent.setup();
    open();
    await waitFor(() => expect(rows()).toHaveLength(2));
    await user.click(within(rows()[0]!).getByRole("button", { name: "预览 seg_01" }));
    const panel = await screen.findByRole("complementary", { name: "段落预览" });
    expect(within(panel).getByLabelText("段落预览视频")).toHaveAttribute("src", "/api/files/previews/abc.mp4");
    expect(panel).toHaveTextContent("seg_01 的预览（v1，新渲染）");
    expect(asked?.searchParams.get("version")).toBe("1");
  });

  it("explains a refused edit (409) and shows the newest plan", async () => {
    let head = plan(1);
    stubApi(
      routes({
        "GET /api/projects/prj_1/plan": () => json(head),
        "POST /api/projects/prj_1/plan:edit": () => {
          head = plan(2);
          return json({ error: { code: "conflict", message: "the plan was updated" } }, 409);
        },
      }),
    );
    const user = userEvent.setup();
    open();
    await waitFor(() => expect(rows()).toHaveLength(2));
    await user.click(within(rows()[0]!).getByRole("button", { name: "锁定镜头 1" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("计划已有新版本，没有改动");
    await waitFor(() => expect(screen.getByLabelText("计划概况")).toHaveTextContent("当前 v2"));
  });
});
