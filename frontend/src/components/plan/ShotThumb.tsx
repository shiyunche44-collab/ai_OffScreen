import type { ShotsView } from "../../api/types";
import { indexAt, spriteStyle } from "../../lib/timeline";

/** The thumbnail of the shot that holds `atMs` (or of shot `shotId`), from the sprite sheets. */
export function ShotThumb({
  view,
  shotId,
  atMs,
  width = 96,
  label,
}: {
  view: ShotsView | undefined;
  shotId?: string | null;
  atMs?: number;
  width?: number;
  label?: string;
}) {
  const shot = view
    ? shotId
      ? view.shots.find((s) => s.id === shotId)
      : atMs !== undefined
        ? view.shots[indexAt(view.shots, atMs)]
        : undefined
    : undefined;
  const style = view && shot ? spriteStyle(view, shot, width) : null;
  return (
    <span
      role="img"
      aria-label={label ?? (shot ? `镜头 ${shot.id}` : "无缩略图")}
      className="inline-block shrink-0 rounded bg-slate-300 dark:bg-slate-700"
      style={style ?? { width, height: Math.round((width * 9) / 16) }}
    />
  );
}
