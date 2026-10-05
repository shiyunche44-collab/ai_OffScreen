import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mergeJobs, useJobEvents } from "../api/events";
import { keys } from "../api/queries";
import type { Job } from "../api/types";
import { FakeEventSource, job, stubEventSource } from "./helpers";

beforeEach(() => stubEventSource());
afterEach(() => vi.unstubAllGlobals());

function setup() {
  const client = new QueryClient();
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
  const view = renderHook(() => useJobEvents("/api/events"), { wrapper });
  const source = FakeEventSource.instances[0]!;
  const jobs = () => client.getQueryData<Job[]>(keys.jobs);
  return { client, view, source, jobs };
}

describe("mergeJobs", () => {
  it("replaces by id and keeps newest first", () => {
    const a = job({ id: "job_a", created_at: "2026-10-05T08:00:00Z" });
    const b = job({ id: "job_b", created_at: "2026-10-05T09:00:00Z" });
    const merged = mergeJobs([a, b], [{ ...a, status: "running", progress: 0.5 }]);
    expect(merged.map((j) => j.id)).toEqual(["job_b", "job_a"]);
    expect(merged[1]).toMatchObject({ status: "running", progress: 0.5 });
  });

  it("starts from nothing", () => {
    expect(mergeJobs(undefined, [job()])).toHaveLength(1);
  });
});

describe("useJobEvents", () => {
  it("connects to the event stream and reports the connection", () => {
    const { view, source } = setup();
    expect(source.url).toBe("/api/events");
    expect(view.result.current).toBe("connecting");
    act(() => source.open());
    expect(view.result.current).toBe("live");
  });

  it("applies the snapshot and then each job event to the jobs query", () => {
    const { source, jobs } = setup();
    act(() => source.emit("snapshot", { jobs: [job({ id: "job_1" }), job({ id: "job_2", created_at: "2026-10-05T09:00:00Z" })] }));
    expect(jobs()?.map((j) => j.id)).toEqual(["job_2", "job_1"]);

    act(() => source.emit("job", job({ id: "job_1", status: "running", progress: 0.25 })));
    expect(jobs()?.find((j) => j.id === "job_1")).toMatchObject({ status: "running", progress: 0.25 });
    act(() => source.emit("job", job({ id: "job_3", created_at: "2026-10-05T10:00:00Z" })));
    expect(jobs()?.map((j) => j.id)).toEqual(["job_3", "job_2", "job_1"]);
  });

  it("keeps jobs known from REST that a snapshot does not mention", () => {
    const { client, source, jobs } = setup();
    client.setQueryData<Job[]>(keys.jobs, [job({ id: "job_old", created_at: "2026-10-01T08:00:00Z" })]);
    act(() => source.emit("snapshot", { jobs: [job({ id: "job_new" })] }));
    expect(jobs()?.map((j) => j.id).sort()).toEqual(["job_new", "job_old"]);
  });

  it("refreshes assets and projects when a job succeeds", () => {
    const { client, source } = setup();
    const spy = vi.spyOn(client, "invalidateQueries");
    act(() => source.emit("job", job({ status: "running" })));
    expect(spy).not.toHaveBeenCalled();
    act(() => source.emit("job", job({ status: "succeeded", progress: 1 })));
    expect(spy).toHaveBeenCalledWith({ queryKey: keys.assets });
    expect(spy).toHaveBeenCalledWith({ queryKey: keys.projects });
  });

  it("refetches the job list after a reconnect, but not on the first connection", () => {
    const { client, source, view } = setup();
    const spy = vi.spyOn(client, "invalidateQueries");
    act(() => source.open());
    expect(spy).not.toHaveBeenCalled();
    act(() => source.fail());
    expect(view.result.current).toBe("connecting"); // the browser is retrying
    act(() => source.open());
    expect(spy).toHaveBeenCalledWith({ queryKey: keys.jobs });
    expect(view.result.current).toBe("live");
  });

  it("closes the stream when the component goes away", () => {
    const { view, source } = setup();
    view.unmount();
    expect(source.closed).toBe(true);
  });
});
