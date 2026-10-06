import { useShots } from "../../api/queries";
import type { Scene } from "../../api/types";
import { formatDuration } from "../../lib/format";
import { fileUrl } from "../../lib/project";

/** One scene: what it is and the stretch of the movie it covers, playable. */
export function ScenePreview({ assetId, scene, onClose }: { assetId: string; scene: Scene | undefined; onClose: () => void }) {
  const shots = useShots(assetId);
  if (!scene) return null;
  const range = `${scene.start_ms / 1000},${scene.end_ms / 1000}`;
  return (
    <aside aria-label="场景预览" className="space-y-2 rounded border border-sky-300 p-3 text-sm dark:border-sky-800">
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <p className="text-xs text-slate-500">
            {scene.id} · {formatDuration(scene.start_ms)}–{formatDuration(scene.end_ms)}
            {scene.location ? ` · ${scene.location}` : ""}
          </p>
          <p className="mt-1 leading-relaxed">{scene.summary}</p>
        </div>
        <button type="button" onClick={onClose} aria-label="关闭预览" className="text-slate-500 hover:text-slate-900">
          ×
        </button>
      </div>
      {shots.data ? (
        <video
          key={scene.id}
          controls
          preload="metadata"
          src={`${fileUrl(shots.data.video)}#t=${range}`}
          aria-label={`${scene.id} 的片段`}
          className="w-full rounded bg-black"
        />
      ) : shots.isPending ? (
        <p className="text-xs text-slate-500">加载视频…</p>
      ) : null}
    </aside>
  );
}
