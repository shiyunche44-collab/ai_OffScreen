import { useCallback, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useAsset, useScenes, useShots, useStory, useTranscript } from "../api/queries";
import { QueryState } from "../components/Query";
import { SceneList } from "../components/SceneList";
import { ShotStrip } from "../components/ShotStrip";
import { TranscriptPanel } from "../components/TranscriptPanel";
import { formatDuration } from "../lib/format";
import { fileUrl } from "../lib/project";

/** A part that may not be built yet: say so instead of failing the page. */
function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section aria-label={title}>
      <h2 className="mb-2 text-lg font-medium">{title}</h2>
      {children}
    </section>
  );
}

export function AnalysisPage() {
  const { assetId = "" } = useParams();
  const asset = useAsset(assetId);
  const shots = useShots(assetId);
  const transcript = useTranscript(assetId);
  const scenes = useScenes(assetId);
  const story = useStory(assetId);

  const video = useRef<HTMLVideoElement>(null);
  const [currentMs, setCurrentMs] = useState(0);
  const seek = useCallback((ms: number) => {
    if (video.current) video.current.currentTime = ms / 1000;
    setCurrentMs(ms);
  }, []);

  return (
    <section>
      <QueryState isPending={asset.isPending} error={asset.error}>
        {asset.data ? (
          <div className="space-y-8">
            <header>
              <p className="text-xs">
                <Link className="text-sky-600 hover:underline dark:text-sky-400" to="/library">
                  ← 素材库
                </Link>
              </p>
              <h1 className="text-xl font-semibold">{asset.data.asset.title}</h1>
              <p className="mt-1 text-xs text-slate-500">
                {formatDuration(asset.data.asset.duration_ms)} · {asset.data.asset.video.width}×
                {asset.data.asset.video.height}
              </p>
            </header>

            <Panel title="播放与镜头">
              <QueryState isPending={shots.isPending} error={shots.error}>
                {shots.data ? (
                  <div className="space-y-3">
                    <video
                      ref={video}
                      controls
                      preload="metadata"
                      src={fileUrl(shots.data.video)}
                      aria-label="代理视频"
                      onTimeUpdate={(e) => setCurrentMs(Math.round(e.currentTarget.currentTime * 1000))}
                      className="w-full max-w-3xl rounded bg-black"
                    />
                    <p className="text-xs text-slate-500">
                      {shots.data.shots.length} 个镜头 · 当前 {formatDuration(currentMs)}
                    </p>
                    <ShotStrip view={shots.data} currentMs={currentMs} onSeek={seek} />
                  </div>
                ) : null}
              </QueryState>
            </Panel>

            <Panel title="剧情">
              <QueryState isPending={story.isPending} error={story.error}>
                {story.data ? (
                  <div className="space-y-2 text-sm leading-relaxed">
                    <p className="font-medium">{story.data.logline}</p>
                    <p>{story.data.synopsis}</p>
                    {story.data.ending ? <p className="text-slate-500">结局：{story.data.ending}</p> : null}
                  </div>
                ) : null}
              </QueryState>
            </Panel>

            <div className="grid gap-8 lg:grid-cols-2">
              <Panel title="台词">
                <QueryState isPending={transcript.isPending} error={transcript.error}>
                  {transcript.data ? (
                    <TranscriptPanel lines={transcript.data.lines} currentMs={currentMs} onSeek={seek} />
                  ) : null}
                </QueryState>
              </Panel>
              <Panel title="场景">
                <QueryState isPending={scenes.isPending} error={scenes.error}>
                  {scenes.data ? (
                    <SceneList scenes={scenes.data.scenes} currentMs={currentMs} onSeek={seek} />
                  ) : null}
                </QueryState>
              </Panel>
            </div>
          </div>
        ) : null}
      </QueryState>
    </section>
  );
}
