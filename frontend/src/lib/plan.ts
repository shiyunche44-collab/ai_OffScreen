import type { Plan, PlanClip, PlanSegment } from "../api/types";
import { fileUrl } from "./project";

/** How long a clip plays: its stretch of the film at its speed. */
export const clipMs = (c: Pick<PlanClip, "src_in_ms" | "src_out_ms" | "speed">): number =>
  Math.round((c.src_out_ms - c.src_in_ms) / (c.speed ?? 1));

/** How long a segment plays: the voice-over if it has one, else its footage. */
export function segmentMs(seg: PlanSegment): number {
  return seg.audio ? seg.audio.duration_ms : seg.clips.reduce((sum, c) => sum + clipMs(c), 0);
}

export const planMs = (plan: Plan): number => plan.segments.reduce((sum, s) => sum + segmentMs(s), 0);

/** Footage minus voice-over, in ms (null for a segment without voice-over). Within a frame or
 * two is normal; the render holds or trims the last clip. */
export function footageGap(seg: PlanSegment): number | null {
  if (!seg.audio) return null;
  return seg.clips.reduce((sum, c) => sum + clipMs(c), 0) - seg.audio.duration_ms;
}

/** URL of a narration segment's audio (plan audio sits next to the plan's versions). */
export function audioUrl(projectId: string, seg: PlanSegment): string | null {
  return seg.audio ? fileUrl(`projects/${projectId}/docs/plan/${seg.audio.file}`) : null;
}

/** 12345 -> "12.345" (trailing zeros dropped), for editing a time in seconds. */
export function secondsText(ms: number): string {
  return (ms / 1000).toFixed(3).replace(/\.?0+$/, "");
}

/** "12.3" -> 12300; null if it is not a non-negative number of seconds. */
export function parseSeconds(text: string): number | null {
  const t = text.trim();
  if (!/^\d+(\.\d+)?$/.test(t)) return null;
  return Math.round(Number(t) * 1000);
}

/** Why an interval of the film cannot be a clip, or null. `limitMs` is the film's length. */
export function intervalProblem(inMs: number | null, outMs: number | null, limitMs?: number): string | null {
  if (inMs === null || outMs === null) return "入点和出点要写成秒数，如 12.5";
  if (inMs >= outMs) return "出点要在入点之后";
  if (limitMs !== undefined && outMs > limitMs) return `出点超出影片（${secondsText(limitMs)} 秒）`;
  return null;
}

export const KIND_LABEL: Record<PlanSegment["kind"], string> = { narration: "解说", original: "原声" };

/** Moving a segment by `delta` places lands on this index, or null at the ends. */
export function movedIndex(index: number, delta: -1 | 1, count: number): number | null {
  const to = index + delta;
  return to < 0 || to >= count ? null : to;
}
