import { useEffect, useRef, useState } from "react";
import type { ShotsView } from "../api/types";
import { formatDuration } from "../lib/format";
import { centeredScroll, indexAt, spriteStyle, visibleWindow } from "../lib/timeline";

const TILE_WIDTH = 120;
const GAP = 4;
const ITEM_WIDTH = TILE_WIDTH + GAP;

/**
 * Every shot as a thumbnail from the sprite sheets, in a horizontal strip. Only the tiles near
 * the viewport exist in the page, so a film with thousands of shots stays light. The shot being
 * played is marked and kept in view; a tile seeks the player.
 */
export function ShotStrip({
  view,
  currentMs,
  onSeek,
}: {
  view: ShotsView;
  currentMs: number;
  onSeek: (ms: number) => void;
}) {
  const box = useRef<HTMLDivElement>(null);
  const [scroll, setScroll] = useState(0);
  const [width, setWidth] = useState(800);
  const current = indexAt(view.shots, currentMs);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const measure = () => setWidth(el.clientWidth || 800);
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  // Follow the playhead, but leave the strip alone while its shot is on screen.
  useEffect(() => {
    const el = box.current;
    if (!el || current < 0) return;
    const left = current * ITEM_WIDTH;
    if (left < el.scrollLeft || left + TILE_WIDTH > el.scrollLeft + el.clientWidth) {
      el.scrollLeft = centeredScroll(current, ITEM_WIDTH, el.clientWidth);
    }
  }, [current]);

  const [first, last] = visibleWindow(scroll, width, ITEM_WIDTH, view.shots.length);
  return (
    <div
      ref={box}
      role="group"
      aria-label="镜头条"
      onScroll={(e) => setScroll(e.currentTarget.scrollLeft)}
      className="overflow-x-auto rounded border border-slate-200 bg-slate-100 dark:border-slate-800 dark:bg-slate-900"
    >
      <div className="relative h-24" style={{ width: view.shots.length * ITEM_WIDTH }}>
        {view.shots.slice(first, last).map((shot, k) => {
          const index = first + k;
          const style = spriteStyle(view, shot, TILE_WIDTH);
          const time = formatDuration(shot.start_ms);
          return (
            <button
              key={shot.id}
              type="button"
              aria-label={`镜头 ${index + 1} ${time}`}
              aria-current={index === current ? "true" : undefined}
              title={`${time}${shot.caption ? ` · ${shot.caption.caption}` : ""}`}
              onClick={() => onSeek(shot.start_ms)}
              className={`absolute top-1 overflow-hidden rounded bg-slate-300 outline-offset-1 dark:bg-slate-700 ${
                index === current ? "outline outline-2 outline-sky-500" : ""
              }`}
              style={{ left: index * ITEM_WIDTH, ...(style ?? { width: TILE_WIDTH, height: 68 }) }}
            >
              <span className="absolute bottom-0 right-0 bg-black/60 px-1 text-[10px] text-white">{time}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
