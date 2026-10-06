import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { keys } from "../api/queries";
import type { Job } from "../api/types";
import { ScriptEditorPage } from "../pages/ScriptEditor";
import { job, json, renderWithProviders, stubApi } from "./helpers";

afterEach(() => vi.unstubAllGlobals());

const PROJECT = {
  project: {
    id: "prj_1",
    asset_id: "ast_1",
    name: "三分钟版",
    options: { minutes: 1, voice: null, style: "suspense", spoil_ending: true },
    created_at: "2026-10-05T08:00:00Z",
  },
  stages: [{ stage: "creation.script", cached: true }],
  video: null,
};

const SCENES = {
  asset_id: "ast_1",
  scenes: [
    { id: "sc_001", start_ms: 0, end_ms: 30_000, shot_ids: ["sh_1"], summary: "女孩与小龙相遇", importance: 0.8 },
    { id: "sc_002", start_ms: 30_000, end_ms: 60_000, shot_ids: ["sh_2"], summary: "雪山分别", importance: 0.6 },
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

const SHOTS = { asset_id: "ast_1", video: "artifacts/analysis.proxy/abc/proxy.mp4", shots: [], sheets: [], columns: 10, rows: 10, tile_width: 160, tile_height: 90 };

const long = (n: number) => "字".repeat(n);

function script(version: number, texts: string[] = [long(130), long(130)], extra: object = {}) {
  return {
    schema_version: 1,
    id: "scr_1",
    project_id: "prj_1",
    version,
    parent_version: version > 1 ? version - 1 : null,
    author: version === 1 ? "ai" : "human",
    params: { style: "suspense", target_duration_s: 60, voice_id: "v", perspective: "third", spoil_ending: true, language: "zh" },
    outline: [
      { beat: "hook", focus: "开场", scene_refs: ["sc_001"], target_s: 30 },
      { beat: "ending", focus: "结尾", scene_refs: ["sc_002"], target_s: 30 },
    ],
    segments: texts.map((text, i) => ({
      id: `seg_0${i + 1}`,
      kind: "narration",
      beat: i === 0 ? "hook" : "ending",
      text,
      scene_refs: [i === 0 ? "sc_001" : "sc_002"],
      line_refs: [],
    })),
    annotations: [{ segment_id: "seg_01", type: "fact_check", message: "资料里没有这件事" }],
    ...extra,
  };
}

const VERSIONS = [
  { kind: "script", version: 2, parent_version: 1, author: "human", created_at: "2026-10-05T09:00:00Z" },
  { kind: "script", version: 1, parent_version: null, author: "ai", created_at: "2026-10-05T08:00:00Z" },
];

type Handler = (r: Request) => Response | Promise<Response>;

function routes(over: Record<string, Handler> = {}): Record<string, Handler> {
  return {
    "GET /api/projects/prj_1": () => json(PROJECT),
    "GET /api/projects/prj_1/script": () => json(script(1)),
    "GET /api/projects/prj_1/script/versions": () => json(VERSIONS.slice(1)),
    "GET /api/projects/prj_1/outline": () => json({ error: { code: "not_found", message: "the outline has not been generated yet" } }, 404),
    "GET /api/assets/ast_1/index/scenes": () => json(SCENES),
    "GET /api/assets/ast_1/index/transcript": () => json(LINES),
    "GET /api/assets/ast_1/index/shots": () => json(SHOTS),
    "GET /api/styles": () =>
      json([
        { id: "roast", name: "轻松吐槽", description: "口语化", tone: "t", structure: [], hook_types: [], perspective: "third", phrases: [], banned_words: [] },
        { id: "suspense", name: "悬疑紧凑", description: "节奏快", tone: "t", structure: [], hook_types: [], perspective: "third", phrases: [], banned_words: [] },
      ]),
    "GET /api/jobs": () => json([]),
    ...over,
  };
}

function open() {
  return renderWithProviders(
    <Routes>
      <Route path="projects/:projectId/script" element={<ScriptEditorPage />} />
    </Routes>,
    "/projects/prj_1/script",
  );
}

const segments = () => within(screen.getByRole("list", { name: "段落" })).getAllByRole("listitem", { name: /^第 \d+ 段$/ });

describe("script editor", () => {
  beforeEach(() => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
  });

  it("shows the segments with counts, the total and the review notes", async () => {
    stubApi(routes());
    open();
    expect(await screen.findByRole("heading", { name: "文案编辑" })).toBeInTheDocument();
    await waitFor(() => expect(segments()).toHaveLength(2));

    expect(screen.getByLabelText("第 1 段文字")).toHaveValue(long(130));
    expect(screen.getByLabelText("字数与时长")).toHaveTextContent("共 2 段 · 260 字 · 预计 0:58 · 目标 1:00");
    expect(screen.getAllByText(/130 字 · 太长/)).toHaveLength(2); // over the 80-character limit
    const notes = within(segments()[0]!).getByRole("list", { name: "审查批注" });
    expect(notes).toHaveTextContent("事实资料里没有这件事");
    expect(within(segments()[1]!).queryByRole("list", { name: "审查批注" })).toBeNull();
  });

  it("previews a scene when its chip is clicked", async () => {
    stubApi(routes());
    const user = userEvent.setup();
    open();
    await waitFor(() => expect(segments()).toHaveLength(2));
    expect(screen.getByText(/点击任何一个场景编号/)).toBeInTheDocument();

    await user.click(within(segments()[1]!).getByRole("button", { name: "sc_002" }));
    const aside = await screen.findByRole("complementary", { name: "场景预览" });
    expect(aside).toHaveTextContent("雪山分别");
    expect(await within(aside).findByLabelText("sc_002 的片段")).toHaveAttribute(
      "src",
      "/api/files/artifacts/analysis.proxy/abc/proxy.mp4#t=30,60",
    );
    await user.click(within(aside).getByRole("button", { name: "关闭预览" }));
    expect(screen.queryByRole("complementary", { name: "场景预览" })).toBeNull();
  });

  it("saves an edit as a new version on the loaded one and drops notes of the edited segment", async () => {
    let body: Record<string, unknown> | undefined;
    let saved = false;
    const v2 = script(2, ["我改过的开场。".repeat(3), long(130)], { annotations: [] });
    stubApi(
      routes({
        "GET /api/projects/prj_1/script": () => json(saved ? v2 : script(1)),
        "PUT /api/projects/prj_1/script": async (r) => {
          body = (await r.json()) as Record<string, unknown>;
          saved = true;
          return json(v2);
        },
      }),
    );
    const user = userEvent.setup();
    open();
    await waitFor(() => expect(segments()).toHaveLength(2));
    expect(screen.getByRole("button", { name: "保存为新版本" })).toBeDisabled();

    const box = screen.getByLabelText("第 1 段文字");
    await user.clear(box);
    await user.type(box, "我改过的开场。我改过的开场。我改过的开场。");
    expect(screen.getByRole("button", { name: "保存为新版本" })).toBeEnabled();
    await user.click(screen.getByRole("button", { name: "保存为新版本" }));

    await waitFor(() => expect(body).toBeDefined());
    expect(body).toMatchObject({ base_version: 1 });
    expect((body!["segments"] as { text: string }[])[0]!.text).toContain("我改过的开场");
    expect(body!["annotations"]).toEqual([]); // the verdict on the old words is not carried over
    await waitFor(() => expect(screen.getByRole("button", { name: "保存为新版本" })).toBeDisabled());
    expect(screen.getByText("当前 v2")).toBeInTheDocument();
  });

  it("cannot save what the backend would refuse, and says why", async () => {
    stubApi(routes());
    const user = userEvent.setup();
    open();
    await waitFor(() => expect(segments()).toHaveLength(2));
    await user.clear(screen.getByLabelText("第 2 段文字"));
    expect(screen.getByRole("button", { name: "保存为新版本" })).toBeDisabled();
    expect(screen.getByRole("list", { name: "文案的问题" })).toHaveTextContent("第 2 段没有文字");
  });

  it("explains a refused save (409) and offers the newest version when the ground moved", async () => {
    stubApi(
      routes({
        "PUT /api/projects/prj_1/script": () => json({ error: { code: "conflict", message: "stale" } }, 409),
      }),
    );
    const user = userEvent.setup();
    const { client } = open();
    await waitFor(() => expect(segments()).toHaveLength(2));
    await user.type(screen.getByLabelText("第 1 段文字"), "追加");

    await user.click(screen.getByRole("button", { name: "保存为新版本" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("文档已被更新，没有保存");

    // meanwhile somebody (an AI job) stored v2: the page keeps the edit and says so
    vi.stubGlobal("fetch", stubFetch(routes({ "GET /api/projects/prj_1/script": () => json(script(2, [long(60), long(60)])) })));
    await client.invalidateQueries({ queryKey: [...keys.project("prj_1"), "script"] });
    const banner = await screen.findByText(/文案已有新版本 v2（你是基于 v1 修改的）/);
    expect(banner).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "载入最新版（放弃我的修改）" }));
    await waitFor(() => expect(screen.getByLabelText("第 1 段文字")).toHaveValue(long(60)));
    expect(screen.queryByText(/文案已有新版本/)).toBeNull();
  });

  it("edits segments: move, insert, delete, scenes and kind", async () => {
    stubApi(routes());
    const user = userEvent.setup();
    open();
    await waitFor(() => expect(segments()).toHaveLength(2));

    await user.click(within(segments()[0]!).getByRole("button", { name: "下移" }));
    expect(segments()[1]).toHaveTextContent("seg_01");

    await user.click(within(segments()[0]!).getByRole("button", { name: "在后面插入一段" }));
    expect(segments()).toHaveLength(3);
    expect(within(segments()[1]!).getByLabelText("第 2 段文字")).toHaveValue("");
    expect(segments()[1]).toHaveTextContent("seg_03");
    expect(screen.getByRole("list", { name: "文案的问题" })).toHaveTextContent("第 2 段没有文字");

    await user.click(within(segments()[1]!).getByRole("button", { name: "删除" }));
    expect(segments()).toHaveLength(2);

    const chips = within(segments()[0]!);
    await user.selectOptions(chips.getByLabelText("添加场景"), "sc_001");
    expect(chips.getByRole("button", { name: "sc_002" })).toBeInTheDocument();
    expect(chips.getByRole("button", { name: "sc_001" })).toBeInTheDocument();
    await user.click(chips.getByRole("button", { name: "移除 sc_002" }));
    expect(chips.queryByRole("button", { name: "sc_002" })).toBeNull();

    await user.selectOptions(chips.getByLabelText("类型"), "original");
    expect(chips.getByText(/原声：播放影片里的这些台词/)).toBeInTheDocument();
    expect(chips.getByRole("list", { name: "可选台词" })).toHaveTextContent("我们不能留在这里。");
    await user.click(chips.getByRole("checkbox", { name: /我们不能留在这里/ }));
    expect(screen.queryByRole("list", { name: "文案的问题" })?.textContent ?? "").not.toContain("第 1 段是原声");
  });

  it("asks the AI to rewrite a segment on the loaded version, and only when nothing is unsaved", async () => {
    let body: Record<string, unknown> | undefined;
    stubApi(
      routes({
        "POST /api/projects/prj_1/script/segments/seg_02:rewrite": async (r) => {
          body = (await r.json()) as Record<string, unknown>;
          return json(job({ id: "job_rw", stage: "creation.rewrite", status: "queued", scope: { asset_id: "ast_1", project_id: "prj_1", segment_id: "seg_02" } }), 202);
        },
      }),
    );
    const user = userEvent.setup();
    open();
    await waitFor(() => expect(segments()).toHaveLength(2));

    await user.type(screen.getByLabelText("第 1 段文字"), "x"); // unsaved edit
    await user.click(within(segments()[1]!).getByRole("button", { name: "AI 改写" }));
    await user.type(screen.getByLabelText("改写指令"), "更口语化");
    expect(screen.getByRole("button", { name: "改写" })).toBeDisabled();
    expect(screen.getByText("先保存或放弃当前的修改，再让 AI 改写")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "放弃修改" }));
    await user.click(screen.getByRole("button", { name: "改写" }));
    await waitFor(() => expect(body).toEqual({ instruction: "更口语化", base_version: 1 }));
    expect(await within(segments()[1]!).findByRole("button", { name: "改写中…" })).toBeDisabled();
  });

  it("shows a failed rewrite on its segment", async () => {
    const failed: Job = job({
      id: "job_rw",
      stage: "creation.rewrite",
      status: "failed",
      error: "Conflict: the script changed while seg_02 was being rewritten",
      scope: { asset_id: "ast_1", project_id: "prj_1", segment_id: "seg_02" },
    });
    stubApi(routes({ "GET /api/jobs": () => json([failed]) }));
    open();
    await waitFor(() => expect(segments()).toHaveLength(2));
    expect(await within(segments()[1]!).findByRole("alert")).toHaveTextContent("改写失败：Conflict");
    expect(within(segments()[0]!).queryByRole("alert")).toBeNull();
  });

  it("lists versions, compares an old one with the current and restores it", async () => {
    let restored: Record<string, unknown> | undefined;
    // version 1 is requested with ?version=1; the page asks for the diff of 1 against 2
    const base = routes({
      "GET /api/projects/prj_1/script": (r) =>
        json(new URL(r.url).searchParams.get("version") === "1" ? script(1) : script(2, [long(60), long(60)])),
      "GET /api/projects/prj_1/script/versions": () => json(VERSIONS),
      "GET /api/projects/prj_1/script/diff": () =>
        json({
          kind: "script",
          a: 1,
          b: 2,
          reordered: false,
          params_changed: ["annotations"],
          changes: [
            { segment_id: "seg_01", status: "changed", fields: ["text"] },
            { segment_id: "seg_02", status: "unchanged", fields: [] },
            { segment_id: "seg_09", status: "added", fields: [] },
          ],
        }),
      "POST /api/projects/prj_1/script:restore": async (r) => {
        restored = (await r.json()) as Record<string, unknown>;
        return json(script(3));
      },
    });
    stubApi(base);
    const user = userEvent.setup();
    open();
    const panel = await screen.findByRole("region", { name: "版本历史" });
    const rows = await within(panel).findAllByRole("listitem");
    expect(rows[0]).toHaveTextContent("v2");
    expect(rows[0]).toHaveTextContent("人工");
    expect(rows[0]).toHaveTextContent("当前");
    expect(rows[1]).toHaveTextContent("v1");
    expect(rows[1]).toHaveTextContent("AI");

    await user.click(within(rows[1]!).getByRole("button", { name: "查看并对比" }));
    const view = await screen.findByLabelText("版本 v1");
    expect(await within(view).findByText("与当前版不同")).toBeInTheDocument();
    expect(view).toHaveTextContent("其他不同：annotations");
    expect(view).toHaveTextContent("当前版本多出的段落：seg_09");
    expect(within(view).getAllByRole("listitem", { name: /^第 \d+ 段$/ })).toHaveLength(2);

    await user.click(within(view).getByRole("button", { name: "恢复这一版（存为新版本）" }));
    await waitFor(() => expect(restored).toEqual({ version: 1, base_version: 2 }));
  });

  it("offers to generate when there is no script yet", async () => {
    let queued = false;
    stubApi(
      routes({
        "GET /api/projects/prj_1/script": () => json({ error: { code: "not_found", message: "the project has no script yet" } }, 404),
        "POST /api/projects/prj_1/script:generate": () => {
          queued = true;
          return json(job({ id: "job_g", stage: "creation.script", scope: { asset_id: "ast_1" } }), 202);
        },
      }),
    );
    const user = userEvent.setup();
    open();
    expect(await screen.findByText(/还没有文案/)).toBeInTheDocument();
    expect(screen.queryByRole("list", { name: "段落" })).toBeNull();
    await user.click(screen.getByRole("button", { name: "生成文案" }));
    await waitFor(() => expect(queued).toBe(true));
    expect(window.confirm).not.toHaveBeenCalled(); // nothing to protect yet
  });

  it("asks before piling a regeneration on an existing script", async () => {
    let queued = 0;
    stubApi(
      routes({
        "POST /api/projects/prj_1/script:generate": () => {
          queued += 1;
          return json(job({ id: "job_g", stage: "creation.script", scope: { asset_id: "ast_1" } }), 202);
        },
      }),
    );
    const user = userEvent.setup();
    open();
    await waitFor(() => expect(segments()).toHaveLength(2));
    vi.mocked(window.confirm).mockReturnValueOnce(false);
    await user.click(screen.getByRole("button", { name: "重新生成文案" }));
    expect(queued).toBe(0);
    await user.click(screen.getByRole("button", { name: "重新生成文案" }));
    await waitFor(() => expect(queued).toBe(1));
  });
});

/** The routes as a plain fetch, for swapping answers in the middle of a test. */
function stubFetch(table: Record<string, Handler>) {
  return async (input: Request | string) => {
    const req = typeof input === "string" ? new Request(new URL(input, "http://localhost")) : input;
    const handler = table[`${req.method} ${new URL(req.url).pathname}`];
    return handler ? handler(req) : json({ error: { code: "not_found", message: "no stub" } }, 404);
  };
}

describe("outline panel", () => {
  const OUTLINE = {
    edited: false,
    total_s: 60,
    outline: {
      schema_version: 1,
      asset_id: "ast_1",
      style: "suspense",
      target_duration_s: 60,
      beats: [
        { beat: "hook", focus: "开场", scene_refs: ["sc_001"], target_s: 20 },
        { beat: "ending", focus: "结尾", scene_refs: ["sc_002"], target_s: 40 },
      ],
    },
  };

  it("offers to generate when there is no outline", async () => {
    let queued = false;
    stubApi(
      routes({
        "POST /api/projects/prj_1/outline:generate": () => {
          queued = true;
          return json(job({ id: "job_o", stage: "creation.outline", scope: { asset_id: "ast_1" } }), 202);
        },
      }),
    );
    const user = userEvent.setup();
    open();
    const panel = await screen.findByRole("region", { name: "大纲" });
    expect(await within(panel).findByText(/还没有大纲/)).toBeInTheDocument();
    await user.click(within(panel).getByRole("button", { name: "生成大纲" }));
    await waitFor(() => expect(queued).toBe(true));
  });

  it("edits beats and saves them; the total is shown against the target", async () => {
    let body: { beats: Record<string, unknown>[] } | undefined;
    stubApi(
      routes({
        "GET /api/projects/prj_1/outline": () => json(OUTLINE),
        "PUT /api/projects/prj_1/outline": async (r) => {
          body = (await r.json()) as typeof body;
          return json({ ...OUTLINE, edited: true, outline: { ...OUTLINE.outline, beats: body!.beats } });
        },
      }),
    );
    const user = userEvent.setup();
    open();
    const panel = await screen.findByRole("region", { name: "大纲" });
    expect(await within(panel).findByText("合计 60 秒 · 目标 60 秒")).toBeInTheDocument();
    const save = within(panel).getByRole("button", { name: "保存大纲" });
    expect(save).toBeDisabled();

    await user.clear(within(panel).getByLabelText("节拍 1 要讲什么"));
    await user.type(within(panel).getByLabelText("节拍 1 要讲什么"), "从结局倒叙");
    await user.clear(within(panel).getByLabelText("节拍 2 秒数"));
    await user.type(within(panel).getByLabelText("节拍 2 秒数"), "30");
    expect(within(panel).getByText("合计 50 秒 · 目标 60 秒")).toBeInTheDocument();
    await user.click(within(panel).getByRole("button", { name: "下移节拍 1" }));
    await user.click(save);

    await waitFor(() => expect(body).toBeDefined());
    expect(body!.beats.map((b) => b["beat"])).toEqual(["ending", "hook"]);
    expect(body!.beats[1]).toMatchObject({ focus: "从结局倒叙", target_s: 20, scene_refs: ["sc_001"] });
    expect(await within(panel).findByText("已手动修改")).toBeInTheDocument();
  });

  it("refuses beats the backend would refuse", async () => {
    stubApi(routes({ "GET /api/projects/prj_1/outline": () => json(OUTLINE) }));
    const user = userEvent.setup();
    open();
    const panel = await screen.findByRole("region", { name: "大纲" });
    await within(panel).findByLabelText("节拍 1 名称");
    const save = within(panel).getByRole("button", { name: "保存大纲" });

    await user.clear(within(panel).getByLabelText("节拍 2 名称"));
    await user.type(within(panel).getByLabelText("节拍 2 名称"), "hook");
    expect(save).toBeDisabled();
    expect(within(panel).getByRole("list", { name: "大纲的问题" })).toHaveTextContent("节拍名「hook」重复");

    await user.clear(within(panel).getByLabelText("节拍 2 名称"));
    await user.type(within(panel).getByLabelText("节拍 2 名称"), "end");
    await user.clear(within(panel).getByLabelText("节拍 2 秒数"));
    await user.type(within(panel).getByLabelText("节拍 2 秒数"), "2");
    expect(within(panel).getByRole("list", { name: "大纲的问题" })).toHaveTextContent("至少 3 秒");
    await user.click(within(panel).getByRole("button", { name: "删除节拍 1" }));
    await user.click(within(panel).getByRole("button", { name: "删除节拍 1" }));
    expect(within(panel).getByRole("list", { name: "大纲的问题" })).toHaveTextContent("大纲不能为空");
    await user.click(within(panel).getByRole("button", { name: "放弃修改" }));
    expect(within(panel).getByLabelText("节拍 1 名称")).toHaveValue("hook");
  });

  it("goes back to the generated outline on request", async () => {
    let edited = true;
    stubApi(
      routes({
        "GET /api/projects/prj_1/outline": () => json({ ...OUTLINE, edited }),
        "DELETE /api/projects/prj_1/outline": () => {
          edited = false;
          return json(OUTLINE);
        },
      }),
    );
    const user = userEvent.setup();
    open();
    const panel = await screen.findByRole("region", { name: "大纲" });
    await user.click(await within(panel).findByRole("button", { name: "恢复为生成版" }));
    await waitFor(() => expect(within(panel).queryByText("已手动修改")).toBeNull());
  });
});

describe("parameters", () => {
  it("saves changed options and says new ones apply to the next generation", async () => {
    let body: Record<string, unknown> | undefined;
    stubApi(
      routes({
        "PATCH /api/projects/prj_1": async (r) => {
          body = (await r.json()) as Record<string, unknown>;
          return json(PROJECT.project);
        },
      }),
    );
    const user = userEvent.setup();
    open();
    const form = await screen.findByRole("form", { name: "参数" });
    const save = within(form).getByRole("button", { name: "保存参数" });
    expect(save).toBeDisabled();
    await within(form).findByRole("option", { name: "轻松吐槽" });

    await user.selectOptions(within(form).getByLabelText("风格"), "roast");
    expect(within(form).getByText("口语化")).toBeInTheDocument();
    await user.clear(within(form).getByLabelText("时长（分钟）"));
    await user.type(within(form).getByLabelText("时长（分钟）"), "2.5");
    await user.click(within(form).getByLabelText("讲出结局"));
    await user.click(save);

    await waitFor(() => expect(body).toBeDefined());
    expect(body).toEqual({
      name: "三分钟版",
      options: { minutes: 2.5, style: "roast", spoil_ending: false, voice: null },
    });
    expect(within(form).getByText(/新参数用于下一次生成/)).toBeInTheDocument();
  });

  it("does not submit a length the backend would refuse", async () => {
    stubApi(routes());
    const user = userEvent.setup();
    open();
    const form = await screen.findByRole("form", { name: "参数" });
    await user.clear(within(form).getByLabelText("时长（分钟）"));
    await user.type(within(form).getByLabelText("时长（分钟）"), "0");
    expect(within(form).getByRole("button", { name: "保存参数" })).toBeDisabled();
  });
});
