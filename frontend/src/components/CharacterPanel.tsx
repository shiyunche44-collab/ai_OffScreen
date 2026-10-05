import { useState } from "react";
import { useBuildCharacters, useCharacters, useEditCharacter, useJobs } from "../api/queries";
import type { Character, CharacterEdit, CharactersView } from "../api/types";
import { isActive } from "../lib/jobs";
import { fileUrl } from "../lib/project";

const SOURCE: Record<string, string> = { ai: "AI 命名", human: "人工命名", tmdb: "演员表" };

function Card({
  character,
  others,
  busy,
  onEdit,
}: {
  character: Character;
  others: Character[];
  busy: boolean;
  onEdit: (change: Partial<CharacterEdit>) => void;
}) {
  const [renaming, setRenaming] = useState(false);
  const [draft, setDraft] = useState(character.name ?? "");
  const thumb = character.face_cluster?.thumbnails[0];
  const label = character.name ?? "未命名";
  return (
    <li
      aria-label={label}
      className="flex gap-3 rounded border border-slate-200 p-3 dark:border-slate-800"
    >
      {thumb ? (
        <img src={fileUrl(thumb)} alt={`${label}的脸`} className="h-20 w-20 shrink-0 rounded object-cover" />
      ) : (
        <div className="h-20 w-20 shrink-0 rounded bg-slate-200 dark:bg-slate-800" />
      )}
      <div className="min-w-0 flex-1 space-y-1 text-sm">
        {renaming ? (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              onEdit({ name: draft.trim() });
              setRenaming(false);
            }}
            className="flex gap-2"
          >
            <input
              aria-label="名字"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              autoFocus
              className="min-w-0 flex-1 rounded border border-slate-300 bg-transparent px-2 py-0.5 dark:border-slate-700"
            />
            <button type="submit" className="rounded bg-sky-600 px-2 text-white">
              保存
            </button>
            <button type="button" onClick={() => setRenaming(false)} className="px-1 text-slate-500">
              取消
            </button>
          </form>
        ) : (
          <div className="flex flex-wrap items-baseline gap-2">
            <span className="font-medium">{label}</span>
            {character.name_source ? (
              <span className="text-xs text-slate-500">{SOURCE[character.name_source] ?? character.name_source}</span>
            ) : null}
            <span className="text-xs text-slate-500">{character.id}</span>
          </div>
        )}
        {character.aliases.length ? <p className="text-xs text-slate-500">又称 {character.aliases.join("、")}</p> : null}
        {character.role ? <p>{character.role}</p> : null}
        {character.bio ? <p className="text-slate-600 dark:text-slate-400">{character.bio}</p> : null}
        <p className="text-xs text-slate-500">{character.face_cluster?.size ?? 0} 张脸</p>
        <div className="flex flex-wrap items-center gap-2 pt-1">
          <button
            type="button"
            disabled={busy}
            onClick={() => {
              setDraft(character.name ?? "");
              setRenaming(true);
            }}
            className="rounded border border-slate-300 px-2 py-0.5 text-xs disabled:opacity-50 dark:border-slate-700"
          >
            改名
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => onEdit({ ignored: true })}
            className="rounded border border-slate-300 px-2 py-0.5 text-xs disabled:opacity-50 dark:border-slate-700"
          >
            忽略
          </button>
          {others.length ? (
            <select
              aria-label={`把${label}合并到`}
              value=""
              disabled={busy}
              onChange={(e) => e.target.value && onEdit({ merged_into: e.target.value })}
              className="rounded border border-slate-300 bg-transparent px-1 py-0.5 text-xs dark:border-slate-700"
            >
              <option value="">合并到…</option>
              {others.map((o) => (
                <option key={o.id} value={o.id}>
                  {o.name ?? o.id}
                </option>
              ))}
            </select>
          ) : null}
          <button
            type="button"
            disabled={busy}
            onClick={() => onEdit({ reset: true })}
            className="rounded px-2 py-0.5 text-xs text-slate-500 disabled:opacity-50"
          >
            还原修改
          </button>
        </div>
      </div>
    </li>
  );
}

function Hidden({ view, busy, onEdit }: { view: CharactersView; busy: boolean; onEdit: (id: string, change: Partial<CharacterEdit>) => void }) {
  const merged = Object.entries(view.merged);
  if (!view.ignored.length && !merged.length && !view.unmatched_edits) return null;
  return (
    <div className="mt-3 space-y-1 text-xs text-slate-500" aria-label="已隐藏的人物">
      {view.ignored.map((id) => (
        <p key={id}>
          已忽略 {id}{" "}
          <button type="button" disabled={busy} onClick={() => onEdit(id, { ignored: false })} className="text-sky-600 hover:underline">
            恢复
          </button>
        </p>
      ))}
      {merged.map(([from, to]) => (
        <p key={from}>
          {from} 已并入 {to}{" "}
          <button type="button" disabled={busy} onClick={() => onEdit(from, { reset: true })} className="text-sky-600 hover:underline">
            拆开
          </button>
        </p>
      ))}
      {view.unmatched_edits ? <p>有 {view.unmatched_edits} 条修改找不到对应的人物（重新聚类后人物变了），仍保留在文件里。</p> : null}
    </div>
  );
}

/** The people of the film: who was found, their names, and the edits a person can make. */
export function CharacterPanel({ assetId }: { assetId: string }) {
  const characters = useCharacters(assetId);
  const build = useBuildCharacters(assetId);
  const edit = useEditCharacter(assetId);
  const jobs = useJobs();
  const active = (jobs.data ?? []).find(
    (j) => j.stage === "analysis.naming" && j.scope["asset_id"] === assetId && isActive(j),
  );
  const failed = !active
    ? (jobs.data ?? []).filter((j) => j.stage === "analysis.naming" && j.scope["asset_id"] === assetId)[0]
    : undefined;

  const trigger = (
    <div className="space-y-2 text-sm">
      {active ? (
        <p className="flex items-center gap-2 text-sky-600">
          识别人物中 {Math.round(active.progress * 100)}%
          <progress className="h-1.5 w-24" max={1} value={active.progress} aria-label="识别进度" />
        </p>
      ) : (
        <>
          <p className="text-slate-500">还没有识别人物。识别会检测人脸、聚类并尝试从台词里找名字（人脸检测较慢）。</p>
          <button
            type="button"
            disabled={build.isPending}
            onClick={() => build.mutate()}
            className="rounded bg-sky-600 px-3 py-1 text-white disabled:opacity-50"
          >
            识别人物
          </button>
          {failed?.status === "failed" ? (
            <p role="alert" className="text-rose-600">
              上次识别失败{failed.error ? `：${failed.error}` : ""}
            </p>
          ) : null}
        </>
      )}
      {build.error ? (
        <p role="alert" className="text-rose-600">
          {build.error.message}
        </p>
      ) : null}
    </div>
  );

  if (characters.isPending) return <p className="text-sm text-slate-500">加载中…</p>;
  if (characters.error || !characters.data) {
    const notBuilt = (characters.error as { status?: number } | null)?.status === 404;
    return notBuilt || active ? (
      trigger
    ) : (
      <p role="alert" className="text-sm text-rose-600">
        出错了：{characters.error?.message}
      </p>
    );
  }

  const view = characters.data;
  const busy = edit.isPending;
  return (
    <div>
      {!view.named ? <p className="mb-2 text-xs text-slate-500">人物已聚类，但还没有命名。</p> : null}
      {active ? trigger : null}
      {view.characters.length ? (
        <ul className="grid gap-3 sm:grid-cols-2">
          {view.characters.map((c) => (
            <Card
              key={c.id}
              character={c}
              others={view.characters.filter((o) => o.id !== c.id)}
              busy={busy}
              onEdit={(change) => edit.mutate({ id: c.id, change })}
            />
          ))}
        </ul>
      ) : (
        <p className="text-sm text-slate-500">没有找到出现足够多次的人物。</p>
      )}
      <Hidden view={view} busy={busy} onEdit={(id, change) => edit.mutate({ id, change })} />
      {edit.error ? (
        <p role="alert" className="mt-2 text-sm text-rose-600">
          {edit.error.message}
        </p>
      ) : null}
    </div>
  );
}
