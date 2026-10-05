import { NavLink, Outlet } from "react-router-dom";
import { useJobEvents, type Connection } from "../api/events";

const nav = [
  { to: "/library", label: "素材库" },
  { to: "/jobs", label: "作业中心" },
];

const badge: Record<Connection, { text: string; className: string }> = {
  live: { text: "实时", className: "bg-emerald-500" },
  connecting: { text: "连接中", className: "bg-amber-500" },
  offline: { text: "已断开", className: "bg-rose-500" },
};

export function Layout() {
  const connection = useJobEvents();
  const { text, className } = badge[connection];
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100">
      <header className="border-b border-slate-200 dark:border-slate-800">
        <div className="mx-auto flex max-w-6xl items-center gap-6 px-4 py-3">
          <span className="font-semibold">AI OffScreen</span>
          <nav className="flex gap-4 text-sm" aria-label="主导航">
            {nav.map(({ to, label }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  isActive ? "font-medium text-sky-600 dark:text-sky-400" : "text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100"
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>
          <span className="ml-auto flex items-center gap-2 text-xs text-slate-500" role="status">
            <span className={`h-2 w-2 rounded-full ${className}`} aria-hidden />
            {text}
          </span>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-6">
        <Outlet />
      </main>
    </div>
  );
}
