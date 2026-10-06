import { useScript, useScriptDiff, useScriptVersions } from "../../api/queries";
import type { DocumentVersion, Scene, ScriptSegment, TranscriptLine } from "../../api/types";
import { SegmentCard } from "./SegmentCard";

const AUTHOR: Record<string, string> = { ai: "AI", human: "人工" };

const when = (iso: string) => new Date(iso).toLocaleString("zh-CN", { hour12: false });

/** The version history, a read-only view of one version and how it differs from the current one. */
export function VersionPanel({
  projectId,
  current,
  selected,
  onSelect,
  scenes,
  lines,
  onPreview,
  onRestore,
  restoring,
  restoreBlocked,
}: {
  projectId: string;
  current: number;
  selected: number | null;
  onSelect: (version: number | null) => void;
  scenes: readonly Scene[] | undefined;
  lines: readonly TranscriptLine[] | undefined;
  onPreview: (sceneId: string) => void;
  onRestore: (version: number) => void;
  restoring: boolean;
  /** Why restoring is not possible right now (unsaved edits). */
  restoreBlocked: string | null;
}) {
  const versions = useScriptVersions(projectId);
  const viewed = useScript(projectId, selected !== null, selected ?? undefined);
  const diff = useScriptDiff(projectId, selected, selected === current ? null : current);
  const marks = new Map(diff.data?.changes.map((c) => [c.segment_id, c.status]));

  // Segments of the viewed version, then the ones only the current version has (they are gone there).
  const shown: { segment: ScriptSegment; mark: "added" | "changed" | "removed" | "unchanged" | undefined }[] = [];
  for (const s of viewed.data?.segments ?? []) {
    const status = marks.get(s.id);
    // Compared "from the viewed version to the current one": added there = missing here.
    shown.push({ segment: s, mark: status === "added" ? undefined : status === "removed" ? "removed" : status });
  }

  return (
    <section aria-label="版本历史" className="space-y-3">
      <h2 className="text-lg font-medium">版本历史</h2>
      {versions.isPending ? <p className="text-sm text-slate-500">加载中…</p> : null}
      {versions.error ? <p className="text-sm text-slate-500">还没有版本。</p> : null}
      <ol className="divide-y divide-slate-200 rounded border border-slate-200 text-sm dark:divide-slate-800 dark:border-slate-800">
        {versions.data?.map((v: DocumentVersion) => (
          <li key={v.version} className="flex flex-wrap items-center gap-x-3 px-3 py-2" aria-current={v.version === current ? "true" : undefined}>
            <span className="font-medium tabular-nums">v{v.version}</span>
            <span className="text-xs text-slate-500">
              {AUTHOR[v.author] ?? v.author}
              {v.parent_version ? ` · 基于 v${v.parent_version}` : ""} · {when(v.created_at)}
            </span>
            {v.version === current ? <span className="rounded bg-emerald-100 px-1.5 text-xs text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200">当前</span> : null}
            <button
              type="button"
              onClick={() => onSelect(selected === v.version ? null : v.version)}
              aria-pressed={selected === v.version}
              className="ml-auto rounded border px-2 py-0.5 text-xs"
            >
              {selected === v.version ? "收起" : v.version === current ? "查看" : "查看并对比"}
            </button>
          </li>
        ))}
      </ol>

      {selected !== null ? (
        <div className="space-y-3" aria-label={`版本 v${selected}`}>
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <span>
              v{selected}
              {selected !== current ? `（与当前 v${current} 对比，标出这一版里不同的段落）` : "（当前版本）"}
            </span>
            {selected !== current ? (
              <button
                type="button"
                disabled={restoring || restoreBlocked !== null}
                title={restoreBlocked ?? undefined}
                onClick={() => onRestore(selected)}
                className="ml-auto rounded border border-sky-500 px-3 py-1 text-sky-700 disabled:opacity-50 dark:text-sky-300"
              >
                恢复这一版（存为新版本）
              </button>
            ) : null}
          </div>
          {diff.data?.reordered ? <p className="text-xs text-amber-600">段落顺序与当前版本不同</p> : null}
          {diff.data && diff.data.params_changed.length > 0 ? (
            <p className="text-xs text-amber-600">其他不同：{diff.data.params_changed.join("、")}</p>
          ) : null}
          {viewed.isPending ? <p className="text-sm text-slate-500">加载中…</p> : null}
          {viewed.error ? <p role="alert" className="text-sm text-rose-600">{viewed.error.message}</p> : null}
          <ol className="space-y-3">
            {shown.map(({ segment, mark }, i) => (
              <SegmentCard
                key={segment.id}
                index={i}
                count={shown.length}
                segment={segment}
                notes={(viewed.data?.annotations ?? []).filter((a) => a.segment_id === segment.id)}
                scenes={scenes}
                lines={lines}
                beats={[]}
                readOnly
                mark={mark}
                onPreview={onPreview}
              />
            ))}
          </ol>
          {diff.data?.changes.some((c) => c.status === "added") ? (
            <p className="text-xs text-slate-500">
              当前版本多出的段落：
              {diff.data.changes.filter((c) => c.status === "added").map((c) => c.segment_id).join("、")}
            </p>
          ) : null}
        </div>
      ) : null}
      {restoreBlocked && selected !== null && selected !== current ? <p className="text-xs text-amber-600">{restoreBlocked}</p> : null}
    </section>
  );
}
