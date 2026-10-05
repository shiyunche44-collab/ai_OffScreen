import createClient from "openapi-fetch";
import type { ErrorBody, paths } from "./types";

/** Every request goes to `/api` on the page's own origin (the Vite proxy in development). */
export const api = createClient<paths>({
  baseUrl: typeof window === "undefined" ? "http://localhost" : window.location.origin,
  // Look `fetch` up per call, so a replaced global (tests) is honoured.
  fetch: (request) => globalThis.fetch(request),
});

/** A failed API call, in the backend's one error shape. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: ErrorBody["details"];

  constructor(status: number, body: ErrorBody | undefined) {
    super(body?.message ?? `request failed (${status})`);
    this.name = "ApiError";
    this.status = status;
    this.code = body?.code ?? "unknown";
    this.details = body?.details;
  }
}

/** The error body of a response; text responses (`parseAs: "text"`) deliver it as a string. */
function errorBody(error: unknown): ErrorBody | undefined {
  let parsed = error;
  if (typeof error === "string") {
    try {
      parsed = JSON.parse(error);
    } catch {
      return undefined;
    }
  }
  return (parsed as { error?: ErrorBody } | undefined)?.error;
}

type Result<T> = { data?: T; error?: unknown; response: Response };

/** The data of a response, or an `ApiError` for anything the API refused. */
export function unwrap<T>({ data, error, response }: Result<T>): T {
  if (response.ok && data !== undefined) return data;
  throw new ApiError(response.status, errorBody(error));
}
