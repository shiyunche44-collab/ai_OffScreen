import { useState } from "react";
import type { PlanSegment, ShotsView } from "../../api/types";
import { formatDuration } from "../../lib/format";
import { KIND_LABEL, audioUrl, clipMs, footageGap, segmentMs } from "../../lib/plan";
import { ShotThumb } from "./ShotThumb";

export interface RowActions {
  swap: (index: number) => void;
  add: () => void;
  trim: (index: number) => void;
  remove: (index: number) => void;
  lock: (index: number, locked: boolean) => void;
  move: (delta: -1 | 1) => void;
  delete: () => void;
  insertOriginal: () => void;
  preview: () => void;
  setVoice: (voiceId: string | null, speed: number | null) => void;
}

/** One segment of the plan: its words, voice-over, and a strip of its clips. */
export function SegmentRow({
  projectId,
  index,
  count,
  segment,
  shots,
  selected,
  busy,
  previewing,
  actions,
}: {
  projectId: string;
  index: number;
  count: number;
  segment: PlanSegment;
  shots: ShotsView | undefined;
  /** The clip whose candidates / trimmer are open, if it is in this segment. */
  selected: number | null;
  busy: boolean;
  previewing: boolean;
  actions: RowActions;
}) {
  const audio = audioUrl(projectId, segment);
  const gap = footageGap(segment);
  const narration = segment.kind === "narration";
  return (
    <li
      aria-label={`第 ${index + 1} 段`}
      className={`space-y-2 rounded border px-4 py-3 ${segment.stale ? "border-amber-400" : "border-slate-200 dark:border-slate-800"}`}
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-500">
        <span className="tabular-nums">
          #{index + 1} · {segment.id}
        </span>
        <span className="rounded bg-slate-100 px-1.5 dark:bg-slate-800">{KIND_LABEL[segment.kind]}</span>
        {segment.stale ? <span className="rounded bg-amber-100 px-1.5 text-amber-700 dark:bg-amber-900 dark:text-amber-200">待重新配音</span> : null}
        <span className="tabular-nums">{formatDuration(segmentMs(segment))}</span>
        {gap !== null && Math.abs(gap) > 80 ? (
          <span className="text-amber-600" title="镜头总长与配音相差较多，渲染时会补足或截断">
            镜头{gap > 0 ? "多" : "少"} {(Math.abs(gap) / 1000).toFixed(1)} 秒
          </span>
        ) : null}
        <span className="ml-auto flex gap-1">
          <button type="button" aria-label={`上移 ${segment.id}`} disabled={busy || index === 0} onClick={() => actions.move(-1)} className="rounded border px-1.5 disabled:opacity-40">
            ↑
          </button>
          <button type="button" aria-label={`下移 ${segment.id}`} disabled={busy || index === count - 1} onClick={() => actions.move(1)} className="rounded border px-1.5 disabled:opacity-40">
            ↓
          </button>
          <button
            type="button"
            aria-label={`删除 ${segment.id}`}
            disabled={busy}
            onClick={() => {
              if (window.confirm(`从计划中删除 ${segment.id}？脚本里的这一段不会变，重建计划也不会把它加回来。`)) actions.delete();
            }}
            className="rounded border px-1.5 text-rose-600 disabled:opacity-40"
          >
            删除
          </button>
        </span>
      </div>

      {narration ? <p className="text-sm leading-relaxed">{segment.text}</p> : <p className="text-sm text-slate-500">播放原片台词{segment.line_refs.length ? `（${segment.line_refs.length} 句）` : ""}</p>}

      {narration && audio ? <audio controls preload="none" src={audio} aria-label={`${segment.id} 配音`} className="h-8 w-full max-w-md" /> : null}

      <ol aria-label="镜头" className="flex flex-wrap gap-2">
        {segment.clips.map((c, i) => (
          <li
            key={`${c.src_in_ms}-${i}`}
            aria-label={`镜头 ${i + 1}`}
            className={`space-y-1 rounded border p-1 ${selected === i ? "border-sky-500" : "border-slate-200 dark:border-slate-800"}`}
          >
            <ShotThumb view={shots} shotId={c.shot_id} atMs={c.src_in_ms} width={96} />
            <p className="text-[11px] tabular-nums text-slate-500">
              {formatDuration(c.src_in_ms)}–{formatDuration(c.src_out_ms)} · {(clipMs(c) / 1000).toFixed(1)}s
              {c.speed !== 1 ? ` · ×${c.speed}` : ""}
              {c.score != null ? ` · ${Math.round(c.score * 100)}分` : ""}
            </p>
            <p className="flex gap-1 text-[11px]">
              <button type="button" aria-pressed={c.locked} aria-label={`${c.locked ? "解锁" : "锁定"}镜头 ${i + 1}`} disabled={busy} onClick={() => actions.lock(i, !c.locked)} className={`rounded border px-1 ${c.locked ? "border-sky-500 text-sky-700 dark:text-sky-300" : ""}`}>
                {c.locked ? "🔒" : "🔓"}
              </button>
              {narration ? (
                <button type="button" aria-label={`替换镜头 ${i + 1}`} disabled={busy || segment.stale} onClick={() => actions.swap(i)} className="rounded border px-1">
                  换镜
                </button>
              ) : null}
              <button type="button" aria-label={`裁剪镜头 ${i + 1}`} disabled={busy} onClick={() => actions.trim(i)} className="rounded border px-1">
                裁剪
              </button>
              {segment.clips.length > 1 ? (
                <button type="button" aria-label={`移除镜头 ${i + 1}`} disabled={busy} onClick={() => actions.remove(i)} className="rounded border px-1 text-rose-600">
                  移除
                </button>
              ) : null}
            </p>
          </li>
        ))}
        {narration ? (
          <li className="self-center">
            <button type="button" disabled={busy || segment.stale} onClick={actions.add} className="rounded border border-dashed px-2 py-1 text-xs">
              ＋ 加镜头
            </button>
          </li>
        ) : null}
      </ol>

      <div className="flex flex-wrap items-center gap-2 text-xs">
        <button type="button" disabled={segment.stale || previewing} onClick={actions.preview} aria-label={`预览 ${segment.id}`} className="rounded border border-sky-500 px-2 py-0.5 text-sky-700 disabled:opacity-50 dark:text-sky-300">
          {previewing ? "渲染中…" : "预览此段"}
        </button>
        <button type="button" disabled={busy} onClick={actions.insertOriginal} aria-label={`在 ${segment.id} 之后插入原声`} className="rounded border px-2 py-0.5">
          在此段后插入原声
        </button>
        {narration && segment.voice ? <VoiceForm segment={segment} busy={busy} onApply={actions.setVoice} /> : null}
      </div>
    </li>
  );
}

function VoiceForm({
  segment,
  busy,
  onApply,
}: {
  segment: PlanSegment;
  busy: boolean;
  onApply: (voiceId: string | null, speed: number | null) => void;
}) {
  const voice = segment.voice!;
  const [voiceId, setVoiceId] = useState(voice.voice_id);
  const [speedText, setSpeedText] = useState(String(voice.speed ?? 1));
  const speed = Number(speedText);
  const speedOk = speed >= 0.5 && speed <= 2;
  const changed = voiceId.trim() !== voice.voice_id || speed !== (voice.speed ?? 1);
  return (
    <details className="ml-auto">
      <summary className="cursor-pointer">
        音色 {voice.voice_id} · 语速 {voice.speed ?? 1}
        {segment.voice_pinned ? " · 已指定" : ""}
      </summary>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <label>
          音色
          <input
            value={voiceId}
            onChange={(e) => setVoiceId(e.target.value)}
            aria-label={`${segment.id} 音色`}
            className="ml-1 w-32 rounded border border-slate-300 bg-transparent px-1 dark:border-slate-700"
          />
        </label>
        <label>
          语速
          <input
            value={speedText}
            onChange={(e) => setSpeedText(e.target.value)}
            aria-label={`${segment.id} 语速`}
            inputMode="decimal"
            className="ml-1 w-16 rounded border border-slate-300 bg-transparent px-1 dark:border-slate-700"
          />
        </label>
        <button
          type="button"
          disabled={busy || !changed || !speedOk || !voiceId.trim()}
          onClick={() => onApply(voiceId.trim() !== voice.voice_id ? voiceId.trim() : null, speed !== (voice.speed ?? 1) ? speed : null)}
          className="rounded border border-sky-500 px-2 py-0.5 text-sky-700 disabled:opacity-50 dark:text-sky-300"
        >
          应用
        </button>
        {!speedOk ? <span className="text-amber-600">语速要在 0.5–2 之间</span> : <span className="text-slate-500">下次「构建计划」时重新配音</span>}
      </div>
    </details>
  );
}
