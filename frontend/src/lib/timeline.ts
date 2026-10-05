import type { ShotsView, ShotView } from "../api/types";
import { fileUrl } from "./project";

interface Span {
  start_ms: number;
  end_ms: number;
}

/** Index of the item whose [start, end) holds `ms` (items sorted, not overlapping); -1 in a gap. */
export function indexAt(items: readonly Span[], ms: number): number {
  let lo = 0;
  let hi = items.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const item = items[mid]!;
    if (ms < item.start_ms) hi = mid - 1;
    else if (ms >= item.end_ms) lo = mid + 1;
    else return mid;
  }
  return -1;
}

/** The half-open range of items to render in a strip scrolled to `scrollLeft`. */
export function visibleWindow(
  scrollLeft: number,
  viewport: number,
  itemWidth: number,
  count: number,
  overscan = 4,
): [number, number] {
  const first = Math.max(0, Math.floor(scrollLeft / itemWidth) - overscan);
  const last = Math.min(count, Math.ceil((scrollLeft + viewport) / itemWidth) + overscan);
  return [first, Math.max(first, last)];
}

/** `scrollLeft` that puts item `index` in the middle of the viewport (never negative). */
export function centeredScroll(index: number, itemWidth: number, viewport: number): number {
  return Math.max(0, index * itemWidth + itemWidth / 2 - viewport / 2);
}

/** CSS that shows the shot's tile of its sprite sheet at `width` px wide, or null without a tile. */
export function spriteStyle(view: ShotsView, shot: ShotView, width: number) {
  const slot = shot.sprite;
  const sheet = slot ? view.sheets[slot.sheet] : undefined;
  if (!slot || !sheet) return null;
  const scale = width / view.tile_width;
  return {
    width,
    height: view.tile_height * scale,
    backgroundImage: `url(${fileUrl(sheet)})`,
    backgroundSize: `${view.tile_width * view.columns * scale}px ${view.tile_height * view.rows * scale}px`,
    backgroundPosition: `-${slot.col * view.tile_width * scale}px -${slot.row * view.tile_height * scale}px`,
  };
}
