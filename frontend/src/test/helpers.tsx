import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";
import type { Job } from "../api/types";

export function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

/** Replace `fetch` with a router of `{"GET /api/jobs": () => Response}` handlers. */
export function stubApi(routes: Record<string, () => Response>) {
  const fetchMock = vi.fn(async (input: Request | string) => {
    const req = typeof input === "string" ? new Request(new URL(input, "http://localhost")) : input;
    const key = `${req.method} ${new URL(req.url).pathname}`;
    const handler = routes[key];
    if (!handler) return json({ error: { code: "not_found", message: `no stub for ${key}` } }, 404);
    return handler();
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

export function renderWithProviders(ui: ReactElement, path = "/") {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const result = render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
  return { client, ...result };
}

export function job(overrides: Partial<Job> = {}): Job {
  return {
    id: "job_1",
    stage: "analysis.story",
    scope: { asset_id: "ast_1" },
    lane: "api",
    status: "queued",
    progress: 0,
    message: "",
    attempt: 0,
    created_at: "2026-10-05T08:00:00Z",
    ...overrides,
  };
}

/** A controllable EventSource: tests push events with `emit`, drop the line with `fail`. */
export class FakeEventSource {
  static instances: FakeEventSource[] = [];
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSED = 2;
  readyState = FakeEventSource.CONNECTING;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;
  private listeners = new Map<string, ((e: MessageEvent<string>) => void)[]>();

  constructor(readonly url: string) {
    FakeEventSource.instances.push(this);
  }
  addEventListener(type: string, fn: (e: MessageEvent<string>) => void) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), fn]);
  }
  close() {
    this.closed = true;
    this.readyState = FakeEventSource.CLOSED;
  }
  open() {
    this.readyState = FakeEventSource.OPEN;
    this.onopen?.();
  }
  emit(type: string, data: unknown) {
    const event = new MessageEvent(type, { data: JSON.stringify(data) });
    for (const fn of this.listeners.get(type) ?? []) fn(event);
  }
  fail(closed = false) {
    this.readyState = closed ? FakeEventSource.CLOSED : FakeEventSource.CONNECTING;
    this.onerror?.();
  }
}

export function stubEventSource() {
  FakeEventSource.instances = [];
  vi.stubGlobal("EventSource", FakeEventSource);
}
