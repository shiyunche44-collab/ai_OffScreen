import { describe, expect, it } from "vitest";
import type { ScriptSegment } from "../api/types";
import {
  contentOf,
  countChars,
  isDirty,
  linesInScenes,
  move,
  newSegmentId,
  problemsOf,
  segmentHint,
  settle,
  toggle,
  totalStatus,
  withKind,
  type ScriptContent,
} from "../lib/script";

const seg = (id: string, text: string, extra: Partial<ScriptSegment> = {}): ScriptSegment => ({
  id,
  kind: "narration",
  text,
  scene_refs: ["sc_001"],
  line_refs: [],
  beat: null,
  ...extra,
});

const content = (segments: ScriptSegment[], annotations: ScriptContent["annotations"] = []): ScriptContent => ({
  params: { style: "suspense", target_duration_s: 60, voice_id: "v" } as ScriptContent["params"],
  outline: [],
  segments,
  annotations,
});

describe("countChars", () => {
  it("counts letters and digits, not punctuation or spaces (like the backend)", () => {
    expect(countChars("这个女孩，为了一条龙！ OK 2 个")).toBe(13);
    expect(countChars("……  ，。")).toBe(0);
  });
});

describe("length hints", () => {
  it("warns about narration outside 15–80 characters", () => {
    expect(segmentHint(seg("a", "字".repeat(14)))).toContain("太短");
    expect(segmentHint(seg("a", "字".repeat(15)))).toBeNull();
    expect(segmentHint(seg("a", "字".repeat(80)))).toBeNull();
    expect(segmentHint(seg("a", "字".repeat(81)))).toContain("太长");
    expect(segmentHint(seg("a", "x", { kind: "original" }))).toBeNull();
  });

  it("judges the total against the target with a 15 % tolerance, narration only", () => {
    // 60 s * 4.5 = 270 characters
    const of = (n: number) => totalStatus([seg("a", "字".repeat(n)), seg("b", "（原声）", { kind: "original" })], 60);
    expect(of(270).state).toBe("ok");
    expect(of(230).state).toBe("ok");
    expect(of(229).state).toBe("short");
    expect(of(310).state).toBe("ok");
    expect(of(311).state).toBe("long");
    expect(of(270)).toMatchObject({ chars: 270, seconds: 60 });
  });
});

describe("editing helpers", () => {
  it("makes unused segment ids", () => {
    expect(newSegmentId([])).toBe("seg_01");
    expect(newSegmentId([{ id: "seg_01" }, { id: "seg_02" }])).toBe("seg_03");
    expect(newSegmentId([{ id: "seg_02" }, { id: "seg_03" }])).toBe("seg_04"); // 3 is taken
  });

  it("moves an item and ignores impossible moves", () => {
    expect(move(["a", "b", "c"], 0, 1)).toEqual(["b", "a", "c"]);
    expect(move(["a", "b", "c"], 2, 0)).toEqual(["c", "a", "b"]);
    expect(move(["a", "b"], 0, 5)).toEqual(["a", "b"]);
    expect(move(["a", "b"], 1, 1)).toEqual(["a", "b"]);
  });

  it("toggles membership without touching the original", () => {
    const a = ["x"];
    expect(toggle(a, "y")).toEqual(["x", "y"]);
    expect(toggle(["x", "y"], "x")).toEqual(["y"]);
    expect(a).toEqual(["x"]);
  });

  it("detects edits by content", () => {
    const a = content([seg("a", "一")]);
    expect(isDirty(a, content([seg("a", "一")]))).toBe(false);
    expect(isDirty(a, content([seg("a", "二")]))).toBe(true);
  });

  it("switches kinds with what each kind needs", () => {
    const toOriginal = withKind(seg("a", "解说词", { scene_refs: ["sc_1"] }), "original");
    expect(toOriginal).toMatchObject({ kind: "original", text: "（原声）", line_refs: [], scene_refs: ["sc_1"] });
    expect(withKind(toOriginal, "narration").text).toBe("");
    const same = seg("a", "x");
    expect(withKind(same, "narration")).toBe(same);
  });

  it("lists the dialogue inside the cited scenes", () => {
    const scenes = [
      { id: "sc_1", start_ms: 0, end_ms: 1000 },
      { id: "sc_2", start_ms: 1000, end_ms: 2000 },
    ];
    const lines = [
      { id: "a", start_ms: 100, end_ms: 200 },
      { id: "b", start_ms: 900, end_ms: 1100 }, // straddles both
      { id: "c", start_ms: 1500, end_ms: 1600 },
    ];
    expect(linesInScenes(lines, scenes, ["sc_1"]).map((l) => l.id)).toEqual(["a", "b"]);
    expect(linesInScenes(lines, scenes, ["sc_2"]).map((l) => l.id)).toEqual(["b", "c"]);
    expect(linesInScenes(lines, scenes, [])).toEqual([]);
  });
});

describe("problemsOf", () => {
  it("names what the backend would refuse", () => {
    expect(problemsOf(content([]))).toEqual(["文案不能为空"]);
    expect(problemsOf(content([seg("a", "  ", { scene_refs: [] })]))).toEqual(["第 1 段没有文字", "第 1 段没有场景引用"]);
    expect(problemsOf(content([seg("a", "x", { kind: "original" })]))).toEqual(["第 1 段是原声，需要选择台词"]);
    expect(problemsOf(content([seg("a", "好"), seg("b", "（原声）", { kind: "original", line_refs: ["ln_1"], scene_refs: [] })]))).toEqual([]);
  });
});

describe("settle", () => {
  const note = (segment_id: string) => ({ segment_id, type: "fact_check" as const, message: "m" });

  it("drops annotations of segments whose words or sources changed, or that are gone", () => {
    const original = content([seg("a", "一"), seg("b", "二"), seg("c", "三"), seg("d", "四")], [note("a"), note("b"), note("c"), note("d")]);
    const draft = content([seg("a", "改了"), seg("b", "二"), seg("c", "三", { scene_refs: ["sc_9"] })], original.annotations);
    expect(settle(draft, original).annotations.map((a) => a.segment_id)).toEqual(["b"]);
  });

  it("keeps everything when nothing changed", () => {
    const original = content([seg("a", "一")], [note("a")]);
    expect(settle(original, original).annotations).toHaveLength(1);
  });

  it("round-trips a script through contentOf", () => {
    const script = { ...content([seg("a", "一")]), id: "scr_1", project_id: "prj_1", version: 3, parent_version: 2, author: "ai", schema_version: 1 };
    expect(contentOf(script as never)).toEqual(content([seg("a", "一")]));
  });
});
