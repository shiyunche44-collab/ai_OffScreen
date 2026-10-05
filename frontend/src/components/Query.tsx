import type { ReactNode } from "react";

/** The loading / error / empty states every list page shares. */
export function QueryState({
  isPending,
  error,
  children,
}: {
  isPending: boolean;
  error: Error | null;
  children: ReactNode;
}) {
  if (isPending) return <p className="text-sm text-slate-500">加载中…</p>;
  if (error) return <p role="alert" className="text-sm text-rose-600">出错了：{error.message}</p>;
  return <>{children}</>;
}
