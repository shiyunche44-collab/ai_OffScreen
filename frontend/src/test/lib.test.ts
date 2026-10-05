import { describe, expect, it } from "vitest";
import { analysisState } from "../lib/analysis";
import { formatDuration, formatSize } from "../lib/format";
import { assetDetail, job } from "./helpers";

describe("format", () => {
  it("formats durations", () => {
    expect(formatDuration(0)).toBe("0:00");
    expect(formatDuration(888032)).toBe("14:48");
    expect(formatDuration(3_723_000)).toBe("1:02:03");
  });

  it("formats sizes", () => {
    expect(formatSize(512)).toBe("512 B");
    expect(formatSize(1536)).toBe("1.5 KB");
    expect(formatSize(681_285_280)).toBe("650 MB");
    expect(formatSize(5 * 1024 ** 3)).toBe("5.0 GB");
  });
});

describe("analysisState", () => {
  const detail = assetDetail();
  const mine = (o: Parameters<typeof job>[0]) => job({ stage: "analysis.story", scope: { asset_id: "ast_1" }, ...o });

  it("is none without jobs", () => {
    expect(analysisState(detail, undefined)).toEqual({ kind: "none" });
    expect(analysisState(detail, [])).toEqual({ kind: "none" });
  });

  it("is done once every stage is cached, whatever the jobs say", () => {
    const done = assetDetail({ cached: [true, true, true, true] });
    expect(analysisState(done, [mine({ status: "running" })])).toEqual({ kind: "done" });
  });

  it("is not done while a stage is missing", () => {
    expect(analysisState(assetDetail({ cached: [true, true, true, false] }), []).kind).toBe("none");
  });

  it("follows the active job", () => {
    expect(analysisState(detail, [mine({ status: "queued" })]).kind).toBe("queued");
    const running = analysisState(detail, [mine({ status: "running", progress: 0.3 })]);
    expect(running.kind).toBe("running");
    expect(running.kind === "running" && running.job.progress).toBe(0.3);
  });

  it("reports a failure, until a newer attempt replaces it", () => {
    const failed = mine({ id: "job_1", status: "failed", error: "boom", created_at: "2026-10-05T08:00:00Z" });
    expect(analysisState(detail, [failed]).kind).toBe("failed");
    const retry = mine({ id: "job_2", status: "queued", created_at: "2026-10-05T09:00:00Z" });
    expect(analysisState(detail, [failed, retry]).kind).toBe("queued");
    const ok = mine({ id: "job_3", status: "succeeded", created_at: "2026-10-05T10:00:00Z" });
    expect(analysisState(detail, [failed, ok]).kind).toBe("none"); // finished but not cached (cleared)
  });

  it("ignores other assets and other stages", () => {
    const other = mine({ status: "running", scope: { asset_id: "ast_2" } });
    const script = mine({ status: "running", stage: "creation.script" });
    expect(analysisState(detail, [other, script]).kind).toBe("none");
  });
});
