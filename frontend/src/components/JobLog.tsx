import { useEffect, useRef } from "react";
import { useJobLog } from "../api/queries";

/** A job's log, following the end while the job is running. */
export function JobLog({ jobId, live }: { jobId: string; live: boolean }) {
  const { data, isPending, error } = useJobLog(jobId, live);
  const box = useRef<HTMLPreElement>(null);

  useEffect(() => {
    if (box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [data]);

  if (isPending) return <p className="text-xs text-slate-500">读取日志…</p>;
  if (error) return <p role="alert" className="text-xs text-rose-600">读取日志失败：{error.message}</p>;
  return (
    <pre
      ref={box}
      aria-label="作业日志"
      className="max-h-64 overflow-auto rounded bg-slate-100 p-3 text-xs leading-relaxed whitespace-pre-wrap dark:bg-slate-900"
    >
      {data ? data : "（还没有日志）"}
    </pre>
  );
}
