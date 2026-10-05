import { useState } from "react";
import { useBrowse } from "../api/queries";
import { formatSize } from "../lib/format";
import { QueryState } from "./Query";

/** Walks the configured media roots; picking a file hands its path to `onPick`. */
export function MediaBrowser({ onPick }: { onPick: (path: string) => void }) {
  const [path, setPath] = useState<string | undefined>(undefined);
  const { data, isPending, error } = useBrowse(path);

  return (
    <div className="rounded border border-slate-200 p-3 text-sm dark:border-slate-800" aria-label="媒体目录">
      <QueryState isPending={isPending} error={error}>
        {data ? (
          <>
            <div className="mb-2 flex items-center gap-2">
              <button
                type="button"
                className="rounded border border-slate-300 px-2 py-0.5 disabled:opacity-40 dark:border-slate-700"
                disabled={data.path === null}
                onClick={() => setPath(data.parent ?? undefined)}
              >
                上一级
              </button>
              <span className="truncate text-xs text-slate-500">{data.path ?? "媒体根目录"}</span>
            </div>
            {data.entries.length === 0 ? (
              <p className="text-slate-500">这里没有文件夹或视频。</p>
            ) : (
              <ul className="max-h-64 divide-y divide-slate-100 overflow-auto dark:divide-slate-900">
                {data.entries.map((entry) => (
                  <li key={entry.path}>
                    {entry.kind === "dir" ? (
                      <button
                        type="button"
                        className="flex w-full items-center gap-2 px-2 py-1.5 text-left hover:bg-slate-100 dark:hover:bg-slate-900"
                        onClick={() => setPath(entry.path)}
                      >
                        <span aria-hidden>📁</span>
                        {entry.name}
                      </button>
                    ) : (
                      <button
                        type="button"
                        className="flex w-full items-center gap-2 px-2 py-1.5 text-left hover:bg-slate-100 disabled:opacity-50 dark:hover:bg-slate-900"
                        disabled={entry.asset_id !== null && entry.asset_id !== undefined}
                        onClick={() => onPick(entry.path)}
                      >
                        <span aria-hidden>🎞</span>
                        <span className="min-w-0 flex-1 truncate">{entry.name}</span>
                        <span className="text-xs text-slate-500">
                          {entry.asset_id ? "已导入" : formatSize(entry.size ?? 0)}
                        </span>
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            )}
            {data.truncated ? <p className="mt-2 text-xs text-amber-600">只显示前 1000 项。</p> : null}
          </>
        ) : null}
      </QueryState>
    </div>
  );
}
