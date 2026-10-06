/** Frame arithmetic for marking cuts. A cut is the number (from 0) of the first frame of a new
 * shot; the player is put in the middle of a frame so seeking never lands on its edge. */

export const fpsOf = (num: number, den: number): number => num / den;

/** The frame a playback position (seconds) is in. */
export const frameAt = (seconds: number, fps: number): number => Math.max(0, Math.floor(seconds * fps + 1e-6));

/** A time inside frame `frame` (its middle). */
export const timeOfFrame = (frame: number, fps: number): number => (frame + 0.5) / fps;

/** The cuts with `frame` added: ascending, no duplicates, never frame 0 (nothing to cut from). */
export function withCut(cuts: readonly number[], frame: number): number[] {
  if (frame < 1 || cuts.includes(frame)) return [...cuts];
  return [...cuts, frame].sort((a, b) => a - b);
}

export const withoutCut = (cuts: readonly number[], frame: number): number[] => cuts.filter((c) => c !== frame);

/** The nearest cut before / after `frame` (null at either end). */
export function neighbours(cuts: readonly number[], frame: number): { prev: number | null; next: number | null } {
  let prev: number | null = null;
  let next: number | null = null;
  for (const c of cuts) {
    if (c < frame) prev = c;
    else if (c > frame && next === null) next = c;
  }
  return { prev, next };
}

/** Cut frames of detected shots: every shot after the first starts at one. */
export function cutsFromShots(startsMs: readonly number[], fps: number): number[] {
  const frames = new Set(startsMs.map((ms) => Math.round((ms * fps) / 1000)).filter((f) => f >= 1));
  return [...frames].sort((a, b) => a - b);
}

export const sameCuts = (a: readonly number[], b: readonly number[]): boolean =>
  a.length === b.length && a.every((c, i) => c === b[i]);
