import { describe, expect, it } from "vitest";
import type { PlanSegment } from "../api/types";
import { audioUrl, clipMs, footageGap, intervalProblem, movedIndex, parseSeconds, secondsText, segmentMs } from "../lib/plan";

const clip = (src_in_ms: number, src_out_ms: number, speed = 1) => ({ asset_id: "ast_1", src_in_ms, src_out_ms, speed, locked: false });

function seg(over: Partial<PlanSegment> = {}): PlanSegment {
  return { id: "seg_01", kind: "narration", stale: false, voice_pinned: false, line_refs: [], clips: [], source_audio: { mode: "duck", stem: "mix", gain_db: 0 }, ...over } as PlanSegment;
}

describe("plan helpers", () => {
  it("a clip plays its stretch of the film divided by its speed", () => {
    expect(clipMs(clip(1000, 4000))).toBe(3000);
    expect(clipMs(clip(1000, 4000, 1.5))).toBe(2000);
  });

  it("a segment lasts as long as its voice-over, or as its footage without one", () => {
    const audio = { file: "tts/a.mp3", duration_ms: 5000, char_timings: [] };
    expect(segmentMs(seg({ audio, clips: [clip(0, 9000)] }))).toBe(5000);
    expect(segmentMs(seg({ kind: "original", clips: [clip(0, 2000), clip(5000, 6000)] }))).toBe(3000);
  });

  it("the footage gap is footage minus voice-over", () => {
    const audio = { file: "tts/a.mp3", duration_ms: 5000, char_timings: [] };
    expect(footageGap(seg({ audio, clips: [clip(0, 3000), clip(10000, 12500)] }))).toBe(500);
    expect(footageGap(seg())).toBeNull();
  });

  it("finds the audio next to the plan's versions", () => {
    const audio = { file: "tts/a b.mp3", duration_ms: 1, char_timings: [] };
    expect(audioUrl("prj_1", seg({ audio }))).toBe("/api/files/projects/prj_1/docs/plan/tts/a%20b.mp3");
    expect(audioUrl("prj_1", seg())).toBeNull();
  });

  it("reads and writes times in seconds to the millisecond", () => {
    expect(secondsText(12345)).toBe("12.345");
    expect(secondsText(12000)).toBe("12");
    expect(secondsText(0)).toBe("0");
    expect(parseSeconds(" 12.5 ")).toBe(12500);
    expect(parseSeconds("1,5")).toBeNull();
    expect(parseSeconds("-1")).toBeNull();
    expect(parseSeconds("")).toBeNull();
  });

  it("names what is wrong with an interval", () => {
    expect(intervalProblem(1000, 2000)).toBeNull();
    expect(intervalProblem(null, 2000)).toMatch(/秒数/);
    expect(intervalProblem(2000, 2000)).toMatch(/之后/);
    expect(intervalProblem(1000, 9000, 5000)).toMatch(/超出影片/);
  });

  it("moves within the list only", () => {
    expect(movedIndex(0, -1, 3)).toBeNull();
    expect(movedIndex(2, 1, 3)).toBeNull();
    expect(movedIndex(1, 1, 3)).toBe(2);
  });
});
