import { useEffect, useState } from "react";
import { useStyles, useUpdateProject } from "../../api/queries";
import type { ProjectDetail } from "../../api/types";
import { optionsOf } from "../../lib/project";

/** The settings a commentary is made with. Changing them rebuilds only the creative steps. */
export function ParamsForm({ detail }: { detail: ProjectDetail }) {
  const styles = useStyles();
  const update = useUpdateProject(detail.project.id);
  const current = optionsOf(detail);
  const [minutes, setMinutes] = useState(String(current.minutes));
  const [style, setStyle] = useState(current.style);
  const [spoil, setSpoil] = useState(current.spoil_ending);
  const [voice, setVoice] = useState(current.voice ?? "");
  const [name, setName] = useState(detail.project.name);

  const key = JSON.stringify([current, detail.project.name]);
  useEffect(() => {
    setMinutes(String(current.minutes));
    setStyle(current.style);
    setSpoil(current.spoil_ending);
    setVoice(current.voice ?? "");
    setName(detail.project.name);
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps

  const minutesNumber = Number(minutes);
  const valid = Number.isFinite(minutesNumber) && minutesNumber > 0 && minutesNumber <= 240 && name.trim() !== "";
  const dirty =
    minutesNumber !== current.minutes ||
    style !== current.style ||
    spoil !== current.spoil_ending ||
    (voice.trim() || null) !== current.voice ||
    name !== detail.project.name;
  const chosen = styles.data?.find((s) => s.id === style);

  return (
    <form
      aria-label="参数"
      className="space-y-3 text-sm"
      onSubmit={(e) => {
        e.preventDefault();
        if (!valid || !dirty) return;
        update.mutate({
          name: name.trim(),
          options: { minutes: minutesNumber, style, spoil_ending: spoil, voice: voice.trim() || null },
        });
      }}
    >
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="space-y-1">
          <span className="block text-xs text-slate-500">项目名</span>
          <input value={name} onChange={(e) => setName(e.target.value)} className="w-full rounded border border-slate-300 bg-transparent px-2 py-1 dark:border-slate-700" />
        </label>
        <label className="space-y-1">
          <span className="block text-xs text-slate-500">时长（分钟）</span>
          <input
            type="number"
            step="0.25"
            min="0.25"
            max="240"
            value={minutes}
            onChange={(e) => setMinutes(e.target.value)}
            className="w-full rounded border border-slate-300 bg-transparent px-2 py-1 dark:border-slate-700"
          />
        </label>
        <div className="space-y-1">
          <label className="block space-y-1">
            <span className="block text-xs text-slate-500">风格</span>
            <select value={style} onChange={(e) => setStyle(e.target.value)} className="w-full rounded border border-slate-300 bg-transparent px-2 py-1 dark:border-slate-700">
              {styles.data?.some((s) => s.id === style) === false || !styles.data ? <option value={style}>{style}</option> : null}
              {styles.data?.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          </label>
          {chosen ? <p className="text-xs text-slate-500">{chosen.description}</p> : null}
        </div>
        <label className="space-y-1">
          <span className="block text-xs text-slate-500">音色（留空用默认）</span>
          <input value={voice} onChange={(e) => setVoice(e.target.value)} className="w-full rounded border border-slate-300 bg-transparent px-2 py-1 dark:border-slate-700" />
        </label>
      </div>
      <label className="flex items-center gap-2">
        <input type="checkbox" checked={spoil} onChange={(e) => setSpoil(e.target.checked)} />
        讲出结局
      </label>
      <div className="flex flex-wrap items-center gap-3">
        <button type="submit" disabled={!dirty || !valid || update.isPending} className="rounded bg-sky-600 px-3 py-1 text-white disabled:opacity-50">
          保存参数
        </button>
        {update.isSuccess && !dirty ? <span className="text-xs text-emerald-600">已保存</span> : null}
        <span className="text-xs text-slate-500">新参数用于下一次生成；已经生成的大纲和文案不会改变。</span>
      </div>
      {update.error ? <p role="alert" className="text-sm text-rose-600">{update.error.message}</p> : null}
    </form>
  );
}
