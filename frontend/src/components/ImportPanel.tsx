import { useState, type FormEvent } from "react";
import { useImportAsset } from "../api/queries";
import { MediaBrowser } from "./MediaBrowser";

export function ImportPanel() {
  const [path, setPath] = useState("");
  const [browsing, setBrowsing] = useState(false);
  const importAsset = useImportAsset();

  function submit(event: FormEvent) {
    event.preventDefault();
    const value = path.trim();
    if (!value) return;
    importAsset.mutate(value, { onSuccess: () => setPath("") });
  }

  return (
    <div className="mb-6 space-y-3">
      <form onSubmit={submit} className="flex gap-2">
        <input
          value={path}
          onChange={(e) => setPath(e.target.value)}
          placeholder="电影文件的完整路径"
          aria-label="电影文件路径"
          className="min-w-0 flex-1 rounded border border-slate-300 bg-white px-3 py-1.5 text-sm dark:border-slate-700 dark:bg-slate-900"
        />
        <button
          type="button"
          onClick={() => setBrowsing((v) => !v)}
          aria-expanded={browsing}
          className="rounded border border-slate-300 px-3 py-1.5 text-sm dark:border-slate-700"
        >
          浏览…
        </button>
        <button
          type="submit"
          disabled={!path.trim() || importAsset.isPending}
          className="rounded bg-sky-600 px-4 py-1.5 text-sm text-white disabled:opacity-50"
        >
          {importAsset.isPending ? "导入中…" : "导入"}
        </button>
      </form>
      {importAsset.error ? (
        <p role="alert" className="text-sm text-rose-600">
          导入失败：{importAsset.error.message}
        </p>
      ) : null}
      {importAsset.isSuccess ? (
        <p role="status" className="text-sm text-emerald-600">
          已导入：{importAsset.data.title}
        </p>
      ) : null}
      {browsing ? <MediaBrowser onPick={setPath} /> : null}
    </div>
  );
}
