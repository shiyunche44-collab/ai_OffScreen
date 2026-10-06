import type { Scene } from "../../api/types";

/**
 * Scene references as chips. A chip previews its scene when clicked; with `onChange` the list is
 * editable (remove a chip, add another scene from the menu).
 */
export function SceneChips({
  ids,
  scenes,
  onPreview,
  onChange,
  label = "场景引用",
}: {
  ids: readonly string[];
  scenes: readonly Scene[] | undefined;
  onPreview: (sceneId: string) => void;
  onChange?: (ids: string[]) => void;
  label?: string;
}) {
  const known = new Set((scenes ?? []).map((s) => s.id));
  const addable = (scenes ?? []).filter((s) => !ids.includes(s.id));
  return (
    <ul aria-label={label} className="flex flex-wrap items-center gap-1.5">
      {ids.map((id) => (
        <li
          key={id}
          className={`flex items-center rounded-full border text-xs ${
            known.size === 0 || known.has(id)
              ? "border-slate-300 dark:border-slate-700"
              : "border-rose-400 text-rose-600"
          }`}
        >
          <button
            type="button"
            onClick={() => onPreview(id)}
            title={known.has(id) ? "预览场景" : "影片里没有这个场景"}
            className="rounded-l-full px-2 py-0.5 hover:bg-slate-100 dark:hover:bg-slate-800"
          >
            {id}
          </button>
          {onChange ? (
            <button
              type="button"
              aria-label={`移除 ${id}`}
              onClick={() => onChange(ids.filter((x) => x !== id))}
              className="rounded-r-full px-1.5 py-0.5 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800"
            >
              ×
            </button>
          ) : (
            <span className="pr-1" />
          )}
        </li>
      ))}
      {onChange && addable.length > 0 ? (
        <li>
          <select
            aria-label="添加场景"
            value=""
            onChange={(e) => e.target.value && onChange([...ids, e.target.value])}
            className="rounded border border-slate-300 bg-transparent px-1 py-0.5 text-xs dark:border-slate-700"
          >
            <option value="">＋ 场景</option>
            {addable.map((s) => (
              <option key={s.id} value={s.id}>
                {s.id} {s.summary.slice(0, 14)}
              </option>
            ))}
          </select>
        </li>
      ) : null}
    </ul>
  );
}
