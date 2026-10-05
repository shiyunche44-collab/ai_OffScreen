import { useQueryClient, type QueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { keys } from "./queries";
import type { Job } from "./types";

export type Connection = "connecting" | "live" | "offline";

/** The shape of the `snapshot` event (see the /api/events description in the OpenAPI document). */
type Snapshot = { jobs: Job[] };

/** Put `incoming` into a newest-first job list, replacing the job with the same id. */
export function mergeJobs(current: Job[] | undefined, incoming: Job[]): Job[] {
  const byId = new Map((current ?? []).map((j) => [j.id, j]));
  for (const job of incoming) byId.set(job.id, job);
  return [...byId.values()].sort((a, b) =>
    a.created_at === b.created_at ? b.id.localeCompare(a.id) : b.created_at.localeCompare(a.created_at),
  );
}

function applyJobs(client: QueryClient, incoming: Job[]): void {
  client.setQueryData<Job[]>(keys.jobs, (current) => mergeJobs(current, incoming));
  // A finished job changes what is built: asset and project status come from the cache state.
  if (incoming.some((j) => j.status === "succeeded")) {
    void client.invalidateQueries({ queryKey: keys.assets });
    void client.invalidateQueries({ queryKey: keys.projects });
  }
}

/**
 * Keeps the `["jobs"]` query live from `/api/events`. `EventSource` reconnects by itself; every
 * connection starts with a snapshot, and after a reconnect the REST list is refetched too, so
 * nothing that happened while offline is lost.
 */
export function useJobEvents(url = "/api/events"): Connection {
  const client = useQueryClient();
  const [state, setState] = useState<Connection>("connecting");

  useEffect(() => {
    const source = new EventSource(url);
    let opened = false;
    source.onopen = () => {
      setState("live");
      if (opened) void client.invalidateQueries({ queryKey: keys.jobs });
      opened = true;
    };
    source.onerror = () => setState(source.readyState === EventSource.CLOSED ? "offline" : "connecting");
    source.addEventListener("snapshot", (e) => {
      applyJobs(client, (JSON.parse((e as MessageEvent<string>).data) as Snapshot).jobs);
    });
    source.addEventListener("job", (e) => {
      applyJobs(client, [JSON.parse((e as MessageEvent<string>).data) as Job]);
    });
    return () => source.close();
  }, [client, url]);

  return state;
}
