import { useState } from "react";
import type { Scene, ScriptSegment, TranscriptLine } from "../../api/types";
import { formatDuration } from "../../lib/format";
import {
  BEATS,
  KIND_LABEL,
  NOTE_LABEL,
  countChars,
  linesInScenes,
  segmentHint,
  toggle,
  withKind,
  type ScriptContent,
} from "../../lib/script";
import { SceneChips } from "./SceneChips";

export type RewriteState = { kind: "idle" } | { kind: "busy" } | { kind: "failed"; message: string };

export function SegmentCard({
  index,
  count,
  segment,
  notes,
  scenes,
  lines,
  beats,
  readOnly = false,
  mark,
  onChange,
  onMove,
  onDelete,
  onInsertAfter,
  onPreview,
  rewrite,
}: {
  index: number;
  count: number;
  segment: ScriptSegment;
  notes: ScriptContent["annotations"];
  scenes: readonly Scene[] | undefined;
  lines: readonly TranscriptLine[] | undefined;
  beats: readonly string[];
  readOnly?: boolean;
  /** How this segment differs in a comparison view. */
  mark?: "added" | "changed" | "removed" | "unchanged";
  onChange?: (segment: ScriptSegment) => void;
  onMove?: (delta: -1 | 1) => void;
  onDelete?: () => void;
  onInsertAfter?: () => void;
  onPreview: (sceneId: string) => void;
  rewrite?: { blocked: string | null; state: RewriteState; submit: (instruction: string) => void };
}) {
  const [asking, setAsking] = useState(false);
  const [instruction, setInstruction] = useState("");
  const edit = (change: Partial<ScriptSegment>) => onChange?.({ ...segment, ...change });
  const hint = segmentHint(segment);
  const beatOptions = segment.beat && !beats.includes(segment.beat) ? [segment.beat, ...beats] : beats;
  const choices = linesInScenes(lines ?? [], scenes ?? [], segment.scene_refs);
  const frame =
    mark === "added"
      ? "border-emerald-400"
      : mark === "changed"
        ? "border-amber-400"
        : mark === "removed"
          ? "border-rose-400 opacity-70"
          : "border-slate-200 dark:border-slate-800";

  return (
    <li className={`space-y-2 rounded border px-4 py-3 ${frame}`} aria-label={`第 ${index + 1} 段`}>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500">
        <span className="tabular-nums">
          #{index + 1} · {segment.id}
        </span>
        {mark && mark !== "unchanged" ? (
          <span className="rounded bg-slate-100 px-1.5 dark:bg-slate-800">
            {{ added: "新增", changed: "与当前版不同", removed: "当前版已删除" }[mark]}
          </span>
        ) : null}
        {readOnly ? (
          <span>
            {BEATS[segment.beat ?? ""] ?? segment.beat ?? ""} {KIND_LABEL[segment.kind]}
          </span>
        ) : (
          <>
            <label className="flex items-center gap-1">
              节拍
              <select
                value={segment.beat ?? ""}
                onChange={(e) => edit({ beat: e.target.value || null })}
                className="rounded border border-slate-300 bg-transparent px-1 dark:border-slate-700"
              >
                <option value="">（无）</option>
                {beatOptions.map((b) => (
                  <option key={b} value={b}>
                    {BEATS[b] ?? b}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-1">
              类型
              <select
                value={segment.kind}
                onChange={(e) => onChange?.(withKind(segment, e.target.value as ScriptSegment["kind"]))}
                className="rounded border border-slate-300 bg-transparent px-1 dark:border-slate-700"
              >
                <option value="narration">{KIND_LABEL.narration}</option>
                <option value="original">{KIND_LABEL.original}</option>
              </select>
            </label>
          </>
        )}
        {segment.kind === "narration" ? (
          <span className={`ml-auto tabular-nums ${hint ? "text-amber-600" : ""}`}>
            {countChars(segment.text)} 字{hint ? ` · ${hint}` : ""}
          </span>
        ) : null}
      </div>

      {segment.kind === "narration" ? (
        readOnly ? (
          <p className="leading-relaxed">{segment.text}</p>
        ) : (
          <textarea
            aria-label={`第 ${index + 1} 段文字`}
            value={segment.text}
            rows={Math.max(2, Math.ceil(segment.text.length / 40))}
            onChange={(e) => edit({ text: e.target.value })}
            className="w-full rounded border border-slate-300 bg-transparent px-2 py-1 leading-relaxed dark:border-slate-700"
          />
        )
      ) : (
        <div className="text-sm">
          <p className="text-slate-500">原声：播放影片里的这些台词</p>
          {readOnly ? (
            <p>{segment.line_refs.join("、") || "（未选择）"}</p>
          ) : segment.scene_refs.length === 0 ? (
            <p className="text-amber-600">先在下面选择场景，再从场景的台词里挑选</p>
          ) : choices.length === 0 ? (
            <p className="text-slate-500">所选场景里没有台词</p>
          ) : (
            <ul aria-label="可选台词" className="mt-1 space-y-1">
              {choices.map((l) => (
                <li key={l.id}>
                  <label className="flex gap-2">
                    <input
                      type="checkbox"
                      checked={segment.line_refs.includes(l.id)}
                      onChange={() => edit({ line_refs: toggle(segment.line_refs, l.id) })}
                    />
                    <span className="tabular-nums text-xs text-slate-500">{formatDuration(l.start_ms)}</span>
                    <span>{l.text}</span>
                  </label>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      <SceneChips
        ids={segment.scene_refs}
        scenes={scenes}
        onPreview={onPreview}
        onChange={readOnly ? undefined : (ids) => edit({ scene_refs: ids })}
      />

      {notes.length > 0 ? (
        <ul aria-label="审查批注" className="space-y-1 text-sm">
          {notes.map((n, i) => (
            <li
              key={i}
              className={`rounded px-2 py-1 ${
                n.type === "fact_check"
                  ? "bg-rose-50 text-rose-800 dark:bg-rose-950 dark:text-rose-200"
                  : "bg-amber-50 text-amber-800 dark:bg-amber-950 dark:text-amber-200"
              }`}
            >
              <span className="mr-2 text-xs font-medium">{NOTE_LABEL[n.type] ?? n.type}</span>
              {n.message}
            </li>
          ))}
        </ul>
      ) : null}

      {!readOnly ? (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <button type="button" onClick={() => onMove?.(-1)} disabled={index === 0} aria-label="上移" className="rounded border px-2 py-0.5 disabled:opacity-40">
            ↑
          </button>
          <button type="button" onClick={() => onMove?.(1)} disabled={index === count - 1} aria-label="下移" className="rounded border px-2 py-0.5 disabled:opacity-40">
            ↓
          </button>
          <button type="button" onClick={onInsertAfter} className="rounded border px-2 py-0.5">
            在后面插入一段
          </button>
          <button type="button" onClick={onDelete} className="rounded border px-2 py-0.5 text-rose-600">
            删除
          </button>
          {rewrite && segment.kind === "narration" ? (
            <button
              type="button"
              onClick={() => setAsking((v) => !v)}
              disabled={rewrite.state.kind === "busy"}
              className="rounded border border-sky-400 px-2 py-0.5 text-sky-700 disabled:opacity-50 dark:text-sky-300"
            >
              {rewrite.state.kind === "busy" ? "改写中…" : "AI 改写"}
            </button>
          ) : null}
        </div>
      ) : null}

      {rewrite?.state.kind === "failed" ? (
        <p role="alert" className="text-xs text-rose-600">
          改写失败：{rewrite.state.message}
        </p>
      ) : null}

      {asking && rewrite ? (
        <form
          className="flex flex-wrap items-center gap-2 text-sm"
          onSubmit={(e) => {
            e.preventDefault();
            if (!instruction.trim() || rewrite.blocked) return;
            rewrite.submit(instruction.trim());
            setAsking(false);
            setInstruction("");
          }}
        >
          <input
            aria-label="改写指令"
            value={instruction}
            maxLength={200}
            placeholder="怎么改？例如：更口语化、加一点悬念、再短一些"
            onChange={(e) => setInstruction(e.target.value)}
            className="min-w-0 flex-1 rounded border border-slate-300 bg-transparent px-2 py-1 dark:border-slate-700"
          />
          <button
            type="submit"
            disabled={!instruction.trim() || rewrite.blocked !== null}
            className="rounded bg-sky-600 px-3 py-1 text-white disabled:opacity-50"
          >
            改写
          </button>
          {rewrite.blocked ? <span className="w-full text-xs text-amber-600">{rewrite.blocked}</span> : null}
        </form>
      ) : null}
    </li>
  );
}
