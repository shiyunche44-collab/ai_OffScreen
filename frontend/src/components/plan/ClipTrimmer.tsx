import { useEffect, useRef, useState } from "react";
import type { PlanClip, ShotsView } from "../../api/types";
import { fileUrl } from "../../lib/project";
import { clipMs, intervalProblem, parseSeconds, secondsText } from "../../lib/plan";

/**
 * Cut one clip on the proxy film: play from the in-point, take the current time as the new
 * in- or out-point, or type seconds. Nothing is stored until "应用".
 */
export function ClipTrimmer({
  segmentId,
  index,
  clip,
  shots,
  filmMs,
  busy,
  onApply,
  onClose,
}: {
  segmentId: string;
  index: number;
  clip: PlanClip;
  shots: ShotsView;
  filmMs: number;
  busy: boolean;
  onApply: (inMs: number, outMs: number, speed: number) => void;
  onClose: () => void;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const [inText, setInText] = useState(secondsText(clip.src_in_ms));
  const [outText, setOutText] = useState(secondsText(clip.src_out_ms));
  const [speedText, setSpeedText] = useState(String(clip.speed ?? 1));
  useEffect(() => {
    setInText(secondsText(clip.src_in_ms));
    setOutText(secondsText(clip.src_out_ms));
    setSpeedText(String(clip.speed ?? 1));
  }, [clip.src_in_ms, clip.src_out_ms, clip.speed, segmentId, index]);

  const inMs = parseSeconds(inText);
  const outMs = parseSeconds(outText);
  const speed = Number(speedText);
  const problem =
    intervalProblem(inMs, outMs, filmMs) ?? (!(speed > 0) ? "速度要大于 0" : null);
  const playMs = inMs !== null && outMs !== null && outMs > inMs && speed > 0 ? clipMs({ src_in_ms: inMs, src_out_ms: outMs, speed }) : null;
  const here = () => Math.round((video.current?.currentTime ?? 0) * 1000);
  const unchanged = inMs === clip.src_in_ms && outMs === clip.src_out_ms && speed === (clip.speed ?? 1);

  return (
    <aside aria-label="裁剪镜头" className="space-y-2 rounded border border-sky-300 p-3 text-sm dark:border-sky-800">
      <div className="flex items-start gap-2">
        <p className="min-w-0 flex-1 text-xs text-slate-500">
          裁剪 {segmentId} 的第 {index + 1} 个镜头
        </p>
        <button type="button" onClick={onClose} aria-label="关闭裁剪" className="text-slate-500 hover:text-slate-900">
          ×
        </button>
      </div>
      <video
        ref={video}
        key={`${segmentId}-${index}`}
        controls
        preload="metadata"
        src={`${fileUrl(shots.video)}#t=${clip.src_in_ms / 1000},${clip.src_out_ms / 1000}`}
        aria-label="裁剪预览"
        className="w-full rounded bg-black"
      />
      <div className="grid grid-cols-[auto_1fr_auto] items-center gap-x-2 gap-y-1">
        <label htmlFor="trim-in">入点（秒）</label>
        <input
          id="trim-in"
          value={inText}
          onChange={(e) => setInText(e.target.value)}
          inputMode="decimal"
          className="rounded border border-slate-300 bg-transparent px-2 py-0.5 tabular-nums dark:border-slate-700"
        />
        <button type="button" onClick={() => setInText(secondsText(here()))} className="rounded border px-2 py-0.5 text-xs">
          取当前时间
        </button>
        <label htmlFor="trim-out">出点（秒）</label>
        <input
          id="trim-out"
          value={outText}
          onChange={(e) => setOutText(e.target.value)}
          inputMode="decimal"
          className="rounded border border-slate-300 bg-transparent px-2 py-0.5 tabular-nums dark:border-slate-700"
        />
        <button type="button" onClick={() => setOutText(secondsText(here()))} className="rounded border px-2 py-0.5 text-xs">
          取当前时间
        </button>
        <label htmlFor="trim-speed">速度</label>
        <input
          id="trim-speed"
          value={speedText}
          onChange={(e) => setSpeedText(e.target.value)}
          inputMode="decimal"
          className="rounded border border-slate-300 bg-transparent px-2 py-0.5 tabular-nums dark:border-slate-700"
        />
        <span />
      </div>
      <p className="text-xs text-slate-500">
        {playMs !== null ? `播放 ${(playMs / 1000).toFixed(2)} 秒` : ""}
        {problem ? <span role="alert" className="text-amber-600"> {problem}</span> : null}
      </p>
      <div className="flex justify-end gap-2">
        <button type="button" onClick={onClose} className="rounded border px-3 py-1 text-sm">
          取消
        </button>
        <button
          type="button"
          disabled={problem !== null || unchanged || busy}
          onClick={() => inMs !== null && outMs !== null && onApply(inMs, outMs, speed)}
          className="rounded bg-sky-600 px-3 py-1 text-sm text-white disabled:opacity-50"
        >
          应用（并锁定）
        </button>
      </div>
    </aside>
  );
}
