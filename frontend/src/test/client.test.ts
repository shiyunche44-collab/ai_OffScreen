import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, unwrap } from "../api/client";
import { json, stubApi } from "./helpers";

afterEach(() => vi.unstubAllGlobals());

describe("typed client", () => {
  it("calls the /api paths with path parameters filled in", async () => {
    const fetchMock = stubApi({
      "GET /api/assets/ast_1": () => json({ asset: { id: "ast_1" }, stages: [] }),
      "POST /api/jobs/job_1:cancel": () => json({ id: "job_1", status: "canceled" }),
    });
    const detail = unwrap(await api.GET("/api/assets/{asset_id}", { params: { path: { asset_id: "ast_1" } } }));
    expect(detail.stages).toEqual([]);
    await api.POST("/api/jobs/{job_id}:cancel", { params: { path: { job_id: "job_1" } } });
    const urls = fetchMock.mock.calls.map(([r]) => (r as Request).url);
    expect(urls.some((u) => u.endsWith("/api/assets/ast_1"))).toBe(true);
    expect(urls.some((u) => u.endsWith("/api/jobs/job_1:cancel"))).toBe(true);
  });

  it("turns the backend's error shape into an ApiError", async () => {
    stubApi({
      "POST /api/assets": () =>
        json({ error: { code: "invalid_input", message: "outside the media roots", details: [{ loc: ["body", "path"] }] } }, 422),
    });
    const result = await api.POST("/api/assets", { body: { path: "/etc/passwd" } });
    const thrown = (() => {
      try {
        unwrap(result);
      } catch (e) {
        return e;
      }
    })();
    expect(thrown).toBeInstanceOf(ApiError);
    expect(thrown).toMatchObject({ status: 422, code: "invalid_input", message: "outside the media roots" });
  });

  it("copes with a failure that is not in the error shape", async () => {
    vi.stubGlobal("fetch", async () => new Response("Bad gateway", { status: 502 }));
    const result = await api.GET("/api/jobs");
    expect(() => unwrap(result)).toThrow(expect.objectContaining({ status: 502, code: "unknown" }));
  });
});
