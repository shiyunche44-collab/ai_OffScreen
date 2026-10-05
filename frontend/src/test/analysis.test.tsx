import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { assetDetail, json, renderWithProviders, stubApi, stubEventSource } from "./helpers";

beforeEach(() => stubEventSource());
afterEach(() => vi.unstubAllGlobals());

const shot = (n: number, caption?: string) => ({
  id: `sh_${n}`,
  start_ms: n * 8000,
  end_ms: (n + 1) * 8000,
  quality: { sharpness: 0.5, brightness: 0.5 },
  keyframes: [`artifacts/k/kf/sh_${n}_a.jpg`, `artifacts/k/kf/sh_${n}_b.jpg`, `artifacts/k/kf/sh_${n}_c.jpg`],
  caption: caption ? { shot_id: `sh_${n}`, caption, shot_size: "wide" } : null,
  sprite: { sheet: 0, col: n, row: 0 },
});

const shotsView = {
  asset_id: "ast_1",
  video: "artifacts/p/proxy_540p.mp4",
  tile_width: 160,
  tile_height: 90,
  columns: 10,
  rows: 10,
  sheets: ["artifacts/k/sprites/sheet_000.jpg"],
  shots: [shot(0, "雪山全景"), shot(1), shot(2, "龙在天上飞")],
};

const lines = [
  { id: "ln_1", start_ms: 1000, end_ms: 3000, text: "Where are you going?", words: [] },
  { id: "ln_2", start_ms: 9000, end_ms: 11000, text: "To the mountain.", words: [] },
];

const scenes = [
  { id: "sc_1", start_ms: 0, end_ms: 16000, shot_ids: ["sh_0", "sh_1"], summary: "女孩出发", location: "雪山", importance: 0.7 },
  { id: "sc_2", start_ms: 16000, end_ms: 24000, shot_ids: ["sh_2"], summary: "遇见龙", importance: 0.9 },
];

const story = {
  asset_id: "ast_1",
  logline: "一个女孩寻找她的龙。",
  synopsis: "她翻过雪山，找到了受伤的龙。",
  acts: [{ name: "全片", summary: "x", scene_ids: ["sc_1", "sc_2"] }],
  ending: "告别",
};

const built = {
  "GET /api/jobs": () => json([]),
  "GET /api/assets/ast_1": () => json(assetDetail({ cached: [true, true, true, true] })),
  "GET /api/assets/ast_1/index/shots": () => json(shotsView),
  "GET /api/assets/ast_1/index/transcript": () => json({ asset_id: "ast_1", language: "en", source: "x", lines }),
  "GET /api/assets/ast_1/index/scenes": () => json({ asset_id: "ast_1", scenes }),
  "GET /api/assets/ast_1/index/story": () => json(story),
};

function setTime(video: HTMLVideoElement, seconds: number) {
  Object.defineProperty(video, "currentTime", { value: seconds, writable: true, configurable: true });
  fireEvent.timeUpdate(video);
}

describe("analysis page", () => {
  it("is reached from an analysed asset in the library", async () => {
    stubApi({ ...built, "GET /api/assets": () => json([assetDetail({ cached: [true, true, true, true] })]) });
    renderWithProviders(<App />, "/library");
    await userEvent.click(await screen.findByRole("link", { name: "Sintel" }));
    expect(await screen.findByRole("heading", { name: "Sintel" })).toBeInTheDocument();
  });

  it("shows the proxy video, the story, the dialogue and the scenes", async () => {
    stubApi(built);
    renderWithProviders(<App />, "/library/ast_1");
    const video = await screen.findByLabelText("代理视频");
    expect(video).toHaveAttribute("src", "/api/files/artifacts/p/proxy_540p.mp4");
    expect(await screen.findByText("一个女孩寻找她的龙。")).toBeInTheDocument();
    expect(screen.getByText("结局：告别")).toBeInTheDocument();
    const dialogue = screen.getByRole("list", { name: "台词" });
    expect(within(dialogue).getAllByRole("button")).toHaveLength(2);
    const sceneList = await screen.findByRole("list", { name: "场景" });
    expect(sceneList).toHaveTextContent("女孩出发");
    expect(sceneList).toHaveTextContent("雪山");
    expect(sceneList).toHaveTextContent("重要度 90%");
    expect(screen.getByText(/3 个镜头/)).toBeInTheDocument();
  });

  it("draws the shot strip from the sprite sheet, with the description as a tooltip", async () => {
    stubApi(built);
    renderWithProviders(<App />, "/library/ast_1");
    const strip = await screen.findByRole("group", { name: "镜头条" });
    const tiles = within(strip).getAllByRole("button");
    expect(tiles).toHaveLength(3);
    expect(tiles[0]).toHaveAttribute("title", "0:00 · 雪山全景");
    expect(tiles[1]).toHaveAttribute("title", "0:08");
    expect(tiles[2]!.style.backgroundImage).toContain("/api/files/artifacts/k/sprites/sheet_000.jpg");
    expect(tiles[2]!.style.backgroundPosition).toBe("-240px 0px");
  });

  it("follows the playhead: highlights the current shot, line and scene", async () => {
    stubApi(built);
    renderWithProviders(<App />, "/library/ast_1");
    const video = (await screen.findByLabelText("代理视频")) as HTMLVideoElement;

    setTime(video, 10);
    await waitFor(() => expect(screen.getByRole("button", { name: /镜头 2/ })).toHaveAttribute("aria-current", "true"));
    expect(screen.getByRole("button", { name: /To the mountain/ }).closest("li")).toHaveAttribute("aria-current", "true");
    expect(screen.getByRole("button", { name: /女孩出发/ }).closest("li")).toHaveAttribute("aria-current", "true");

    setTime(video, 20);
    await waitFor(() => expect(screen.getByRole("button", { name: /镜头 3/ })).toHaveAttribute("aria-current", "true"));
    expect(screen.getByRole("button", { name: /遇见龙/ }).closest("li")).toHaveAttribute("aria-current", "true");
    // between lines nothing is highlighted
    expect(screen.getByRole("button", { name: /To the mountain/ }).closest("li")).not.toHaveAttribute("aria-current");
  });

  it("seeks the player from a shot, a line and a scene", async () => {
    stubApi(built);
    renderWithProviders(<App />, "/library/ast_1");
    const video = (await screen.findByLabelText("代理视频")) as HTMLVideoElement;
    Object.defineProperty(video, "currentTime", { value: 0, writable: true, configurable: true });

    await userEvent.click(screen.getByRole("button", { name: /镜头 3/ }));
    expect(video.currentTime).toBe(16);
    await userEvent.click(screen.getByRole("button", { name: /To the mountain/ }));
    expect(video.currentTime).toBe(9);
    await userEvent.click(screen.getByRole("button", { name: /遇见龙/ }));
    expect(video.currentTime).toBe(16);
    expect(screen.getByText(/当前 0:16/)).toBeInTheDocument();
  });

  it("says which part is not built yet while the rest still works", async () => {
    stubApi({
      ...built,
      "GET /api/assets/ast_1/index/scenes": () =>
        json({ error: { code: "not_found", message: "analysis.scenes has not been built yet" } }, 404),
    });
    renderWithProviders(<App />, "/library/ast_1");
    expect(await screen.findByText(/analysis.scenes has not been built yet/)).toBeInTheDocument();
    expect(await screen.findByRole("group", { name: "镜头条" })).toBeInTheDocument();
    expect(screen.getByRole("list", { name: "台词" })).toBeInTheDocument();
  });

  it("only renders the shots near the viewport of a long film", async () => {
    const many = Array.from({ length: 3000 }, (_, i) => ({ ...shot(0), id: `sh_${i}`, start_ms: i * 1000, end_ms: (i + 1) * 1000, sprite: { sheet: 0, col: i % 10, row: 0 } }));
    stubApi({ ...built, "GET /api/assets/ast_1/index/shots": () => json({ ...shotsView, shots: many }) });
    renderWithProviders(<App />, "/library/ast_1");
    const strip = await screen.findByRole("group", { name: "镜头条" });
    const rendered = within(strip).getAllByRole("button").length;
    expect(rendered).toBeGreaterThan(0);
    expect(rendered).toBeLessThan(40);
  });

  it("is an error for an unknown asset", async () => {
    stubApi({
      "GET /api/jobs": () => json([]),
      "GET /api/assets/ast_x": () => json({ error: { code: "not_found", message: "unknown asset ast_x" } }, 404),
      "GET /api/assets/ast_x/index/shots": () => json({ error: { code: "not_found", message: "unknown asset ast_x" } }, 404),
      "GET /api/assets/ast_x/index/transcript": () => json({ error: { code: "not_found", message: "unknown asset ast_x" } }, 404),
      "GET /api/assets/ast_x/index/scenes": () => json({ error: { code: "not_found", message: "unknown asset ast_x" } }, 404),
      "GET /api/assets/ast_x/index/story": () => json({ error: { code: "not_found", message: "unknown asset ast_x" } }, 404),
    });
    renderWithProviders(<App />, "/library/ast_x");
    expect(await screen.findByRole("alert")).toHaveTextContent("unknown asset ast_x");
  });
});

