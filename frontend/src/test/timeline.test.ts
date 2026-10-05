import { describe, expect, it } from "vitest";
import type { ShotsView } from "../api/types";
import { centeredScroll, indexAt, spriteStyle, visibleWindow } from "../lib/timeline";

const spans = [
  { start_ms: 0, end_ms: 1000 },
  { start_ms: 1000, end_ms: 2500 },
  { start_ms: 4000, end_ms: 5000 },
];

describe("indexAt", () => {
  it("finds the item holding the time, ends exclusive", () => {
    expect(indexAt(spans, 0)).toBe(0);
    expect(indexAt(spans, 999)).toBe(0);
    expect(indexAt(spans, 1000)).toBe(1);
    expect(indexAt(spans, 4999)).toBe(2);
    expect(indexAt(spans, 5000)).toBe(-1);
  });
  it("answers -1 in a gap, before the first item and for an empty list", () => {
    expect(indexAt(spans, 3000)).toBe(-1);
    expect(indexAt([{ start_ms: 500, end_ms: 900 }], 100)).toBe(-1);
    expect(indexAt([], 10)).toBe(-1);
  });
  it("agrees with a linear search on many items", () => {
    const many = Array.from({ length: 500 }, (_, i) => ({ start_ms: i * 100, end_ms: i * 100 + 80 }));
    for (let ms = 0; ms < 50_000; ms += 37) {
      expect(indexAt(many, ms)).toBe(many.findIndex((s) => ms >= s.start_ms && ms < s.end_ms));
    }
  });
});

describe("visibleWindow", () => {
  it("covers the viewport plus the overscan, clamped to the list", () => {
    expect(visibleWindow(0, 400, 100, 1000, 2)).toEqual([0, 6]);
    expect(visibleWindow(5000, 400, 100, 1000, 2)).toEqual([48, 56]);
    expect(visibleWindow(99_700, 400, 100, 1000, 2)).toEqual([995, 1000]);
    expect(visibleWindow(0, 400, 100, 0)).toEqual([0, 0]);
  });
});

describe("centeredScroll", () => {
  it("centres an item and never goes negative", () => {
    expect(centeredScroll(10, 100, 400)).toBe(850);
    expect(centeredScroll(0, 100, 400)).toBe(0);
  });
});

describe("spriteStyle", () => {
  const view = {
    asset_id: "ast_1",
    video: "artifacts/p/proxy_540p.mp4",
    tile_width: 160,
    tile_height: 90,
    columns: 10,
    rows: 10,
    sheets: ["artifacts/k/sprites/sheet_000.jpg", "artifacts/k/sprites/sheet_001.jpg"],
    shots: [],
  } as ShotsView;
  const shot = (sheet: number, col: number, row: number) =>
    ({ id: "sh_1", start_ms: 0, end_ms: 1, keyframes: [], sprite: { sheet, col, row } }) as ShotsView["shots"][number];

  it("scales the whole sheet so one tile is `width` wide and offsets to the tile", () => {
    expect(spriteStyle(view, shot(1, 3, 2), 80)).toEqual({
      width: 80,
      height: 45,
      backgroundImage: "url(/api/files/artifacts/k/sprites/sheet_001.jpg)",
      backgroundSize: "800px 450px",
      backgroundPosition: "-240px -90px",
    });
  });
  it("has no style for a shot without a tile or with a missing sheet", () => {
    expect(spriteStyle(view, { ...shot(0, 0, 0), sprite: null }, 80)).toBeNull();
    expect(spriteStyle(view, shot(5, 0, 0), 80)).toBeNull();
  });
});
