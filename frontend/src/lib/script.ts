import type { Script, ScriptSegment } from "../api/types";
import { CHARS_PER_SECOND } from "./project";

/** What a person edits of a script: everything but its identity and history. */
export interface ScriptContent {
  params: Script["params"];
  outline: Script["outline"];
  segments: Script["segments"];
  annotations: Script["annotations"];
}

export const contentOf = (s: Script): ScriptContent => ({
  params: s.params,
  outline: s.outline ?? [],
  segments: s.segments,
  annotations: s.annotations ?? [],
});

export const BEATS: Record<string, string> = {
  hook: "钩子",
  setup: "铺垫",
  development: "发展",
  climax: "高潮",
  ending: "结尾",
};
export const beatLabel = (beat: string | null | undefined) => (beat ? (BEATS[beat] ?? beat) : "");

/** Spoken characters: letters and digits, so punctuation and spaces do not count (as the backend). */
export function countChars(text: string): number {
  let n = 0;
  for (const ch of text) if (/[\p{L}\p{N}]/u.test(ch)) n += 1;
  return n;
}

export const estimateSeconds = (chars: number) => chars / CHARS_PER_SECOND;

/** The segment length the rule checker asks for (ARCHITECTURE §7.2). */
export const MIN_SEGMENT_CHARS = 15;
export const MAX_SEGMENT_CHARS = 80;
/** How far the whole text may stray from the target length. */
export const TOTAL_TOLERANCE = 0.15;

export function segmentHint(segment: Pick<ScriptSegment, "kind" | "text">): string | null {
  if (segment.kind !== "narration") return null;
  const n = countChars(segment.text);
  if (n < MIN_SEGMENT_CHARS) return `太短（${n} 字，至少 ${MIN_SEGMENT_CHARS}）`;
  if (n > MAX_SEGMENT_CHARS) return `太长（${n} 字，至多 ${MAX_SEGMENT_CHARS}）`;
  return null;
}

export type TotalStatus = { chars: number; seconds: number; targetSeconds: number; state: "ok" | "short" | "long" };

export function totalStatus(segments: readonly Pick<ScriptSegment, "kind" | "text">[], targetSeconds: number): TotalStatus {
  const chars = segments.reduce((n, s) => (s.kind === "narration" ? n + countChars(s.text) : n), 0);
  const target = targetSeconds * CHARS_PER_SECOND;
  const state = chars < target * (1 - TOTAL_TOLERANCE) ? "short" : chars > target * (1 + TOTAL_TOLERANCE) ? "long" : "ok";
  return { chars, seconds: estimateSeconds(chars), targetSeconds, state };
}

/** `seg_NN` not used yet. */
export function newSegmentId(segments: readonly { id: string }[]): string {
  const used = new Set(segments.map((s) => s.id));
  let n = segments.length + 1;
  while (used.has(`seg_${String(n).padStart(2, "0")}`)) n += 1;
  return `seg_${String(n).padStart(2, "0")}`;
}

export function move<T>(items: readonly T[], from: number, to: number): T[] {
  if (from === to || from < 0 || to < 0 || from >= items.length || to >= items.length) return [...items];
  const next = [...items];
  const [item] = next.splice(from, 1);
  next.splice(to, 0, item as T);
  return next;
}

/** Edited content differs from what was loaded. Content is plain JSON, so a string compare is exact. */
export const isDirty = (a: ScriptContent, b: ScriptContent) => JSON.stringify(a) !== JSON.stringify(b);

/** Annotations of one segment. */
export const notesOf = (content: ScriptContent, segmentId: string) =>
  content.annotations.filter((a) => a.segment_id === segmentId);

export const NOTE_LABEL: Record<string, string> = { fact_check: "事实", rule: "规则", style: "风格" };

/** After the segments changed, drop annotations of segments that are gone. */
export function pruneNotes(content: ScriptContent): ScriptContent {
  const ids = new Set(content.segments.map((s) => s.id));
  return { ...content, annotations: content.annotations.filter((a) => ids.has(a.segment_id)) };
}

/** Why this content cannot be saved yet (the backend refuses the same things). */
export function problemsOf(content: ScriptContent): string[] {
  const out: string[] = [];
  if (content.segments.length === 0) out.push("文案不能为空");
  content.segments.forEach((s, i) => {
    const n = i + 1;
    if (s.kind === "narration") {
      if (!s.text.trim()) out.push(`第 ${n} 段没有文字`);
      if (s.scene_refs.length === 0) out.push(`第 ${n} 段没有场景引用`);
    } else if (s.line_refs.length === 0) {
      out.push(`第 ${n} 段是原声，需要选择台词`);
    }
  });
  return out;
}

/** The segment kinds a person can switch between. */
export const KIND_LABEL: Record<ScriptSegment["kind"], string> = { narration: "解说", original: "原声" };

export const ORIGINAL_TEXT = "（原声）";

export function withKind(segment: ScriptSegment, kind: ScriptSegment["kind"]): ScriptSegment {
  if (segment.kind === kind) return segment;
  return kind === "original"
    ? { ...segment, kind, text: ORIGINAL_TEXT, line_refs: [] }
    : { ...segment, kind, text: segment.text === ORIGINAL_TEXT ? "" : segment.text, line_refs: [] };
}

/** `a` plus `x` unless already there; `a` without `x`. */
export const toggle = (a: readonly string[], x: string): string[] => (a.includes(x) ? a.filter((y) => y !== x) : [...a, x]);

/** The dialogue that falls inside the given scenes (what an original-sound segment may play). */
export function linesInScenes<L extends { start_ms: number; end_ms: number }>(
  lines: readonly L[],
  scenes: readonly { id: string; start_ms: number; end_ms: number }[],
  sceneIds: readonly string[],
): L[] {
  const spans = scenes.filter((s) => sceneIds.includes(s.id));
  return lines.filter((l) => spans.some((s) => l.start_ms < s.end_ms && l.end_ms > s.start_ms));
}

/**
 * What to store: annotations of segments that are gone, or whose words or sources were changed
 * since `original`, are dropped (they judged text that no longer exists).
 */
export function settle(draft: ScriptContent, original: ScriptContent): ScriptContent {
  const before = new Map(original.segments.map((s) => [s.id, s]));
  const changed = new Set(
    draft.segments
      .filter((s) => {
        const o = before.get(s.id);
        return o !== undefined && (o.text !== s.text || JSON.stringify(o.scene_refs) !== JSON.stringify(s.scene_refs));
      })
      .map((s) => s.id),
  );
  return pruneNotes({ ...draft, annotations: draft.annotations.filter((a) => !changed.has(a.segment_id)) });
}
