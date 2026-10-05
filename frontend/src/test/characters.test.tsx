import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "../App";
import { assetDetail, job, json, renderWithProviders, stubApi, stubEventSource } from "./helpers";

beforeEach(() => stubEventSource());
afterEach(() => vi.unstubAllGlobals());

const character = (id: string, over: Record<string, unknown> = {}) => ({
  id,
  name: null,
  name_source: null,
  aliases: [],
  role: null,
  bio: null,
  face_cluster: { size: 12, centroid_ref: "centroids.npy#0", thumbnails: [`artifacts/n/faces/${id}_1.jpg`] },
  ...over,
});

const view = (over: Record<string, unknown> = {}) => ({
  asset_id: "ast_1",
  characters: [
    character("ch_01", { name: "Sintel", name_source: "ai", aliases: ["the girl"], role: "主角", bio: "寻找她的龙。" }),
    character("ch_02", { face_cluster: { size: 6, centroid_ref: "centroids.npy#1", thumbnails: [] } }),
  ],
  merged: {},
  ignored: [],
  unmatched_edits: 0,
  named: true,
  ...over,
});

const notBuilt = () =>
  json({ error: { code: "not_found", message: "analysis.characters has not been built yet" } }, 404);

const base = {
  "GET /api/jobs": () => json([]),
  "GET /api/assets/ast_1": () => json(assetDetail({ cached: [true, true, true, true] })),
};

function panel() {
  return screen.findByRole("region", { name: "人物" });
}

describe("character panel", () => {
  it("offers to identify the people when nothing is built, and queues the job", async () => {
    const queued = vi.fn(() => json(job({ stage: "analysis.naming", scope: { asset_id: "ast_1" } }), 202));
    stubApi({
      ...base,
      "GET /api/assets/ast_1/index/characters": notBuilt,
      "POST /api/assets/ast_1/characters:build": queued,
    });
    renderWithProviders(<App />, "/library/ast_1");
    const region = await panel();
    expect(await within(region).findByText(/还没有识别人物/)).toBeInTheDocument();
    await userEvent.click(within(region).getByRole("button", { name: "识别人物" }));
    await waitFor(() => expect(queued).toHaveBeenCalled());
  });

  it("shows the progress of a running identification instead of the button", async () => {
    stubApi({
      ...base,
      "GET /api/jobs": () =>
        json([job({ stage: "analysis.naming", status: "running", progress: 0.4, scope: { asset_id: "ast_1" } })]),
      "GET /api/assets/ast_1/index/characters": notBuilt,
    });
    renderWithProviders(<App />, "/library/ast_1");
    const region = await panel();
    expect(await within(region).findByText(/识别人物中 40%/)).toBeInTheDocument();
    expect(within(region).queryByRole("button", { name: "识别人物" })).not.toBeInTheDocument();
  });

  it("reports a failed identification next to the button", async () => {
    stubApi({
      ...base,
      "GET /api/jobs": () =>
        json([job({ stage: "analysis.naming", status: "failed", error: "no face analyzer", scope: { asset_id: "ast_1" } })]),
      "GET /api/assets/ast_1/index/characters": notBuilt,
    });
    renderWithProviders(<App />, "/library/ast_1");
    const region = await panel();
    expect(await within(region).findByRole("alert")).toHaveTextContent("上次识别失败：no face analyzer");
    expect(within(region).getByRole("button", { name: "识别人物" })).toBeEnabled();
  });

  it("lists the characters with their thumbnail, name, source, role and face count", async () => {
    stubApi({ ...base, "GET /api/assets/ast_1/index/characters": () => json(view()) });
    renderWithProviders(<App />, "/library/ast_1");
    const sintel = await screen.findByRole("listitem", { name: "Sintel" });
    expect(sintel).toHaveTextContent("AI 命名");
    expect(sintel).toHaveTextContent("又称 the girl");
    expect(sintel).toHaveTextContent("主角");
    expect(sintel).toHaveTextContent("12 张脸");
    expect(within(sintel).getByRole("img")).toHaveAttribute("src", "/api/files/artifacts/n/faces/ch_01_1.jpg");
    const unnamed = screen.getByRole("listitem", { name: "未命名" });
    expect(unnamed).toHaveTextContent("ch_02");
    expect(within(unnamed).queryByRole("img")).not.toBeInTheDocument();
  });

  it("says when the characters are grouped but not named", async () => {
    stubApi({ ...base, "GET /api/assets/ast_1/index/characters": () => json(view({ named: false })) });
    renderWithProviders(<App />, "/library/ast_1");
    expect(await screen.findByText("人物已聚类，但还没有命名。")).toBeInTheDocument();
  });

  it("renames a character and shows the answer", async () => {
    const patch = vi.fn(async (req: Request) => {
      expect(await req.json()).toEqual({ reset: false, name: "Dragon" });
      return json(
        view({
          characters: [
            character("ch_01", { name: "Sintel", name_source: "ai" }),
            character("ch_02", { name: "Dragon", name_source: "human" }),
          ],
        }),
      );
    });
    stubApi({
      ...base,
      "GET /api/assets/ast_1/index/characters": () => json(view()),
      "PATCH /api/assets/ast_1/characters/ch_02": patch,
    });
    renderWithProviders(<App />, "/library/ast_1");
    const card = await screen.findByRole("listitem", { name: "未命名" });
    await userEvent.click(within(card).getByRole("button", { name: "改名" }));
    await userEvent.type(within(card).getByRole("textbox", { name: "名字" }), "Dragon");
    await userEvent.click(within(card).getByRole("button", { name: "保存" }));
    const renamed = await screen.findByRole("listitem", { name: "Dragon" });
    expect(renamed).toHaveTextContent("人工命名");
    expect(patch).toHaveBeenCalledTimes(1);
  });

  it("can cancel a rename without sending anything", async () => {
    const patch = vi.fn(() => json(view()));
    stubApi({
      ...base,
      "GET /api/assets/ast_1/index/characters": () => json(view()),
      "PATCH /api/assets/ast_1/characters/ch_01": patch,
    });
    renderWithProviders(<App />, "/library/ast_1");
    const card = await screen.findByRole("listitem", { name: "Sintel" });
    await userEvent.click(within(card).getByRole("button", { name: "改名" }));
    expect(within(card).getByRole("textbox", { name: "名字" })).toHaveValue("Sintel");
    await userEvent.click(within(card).getByRole("button", { name: "取消" }));
    expect(within(card).queryByRole("textbox")).not.toBeInTheDocument();
    expect(patch).not.toHaveBeenCalled();
  });

  it("ignores a character, merges one into another, and can bring them back", async () => {
    const bodies: unknown[] = [];
    stubApi({
      ...base,
      "GET /api/assets/ast_1/index/characters": () => json(view()),
      "PATCH /api/assets/ast_1/characters/ch_02": async (req) => {
        const body = await req.json();
        bodies.push(body);
        return json(
          body.merged_into
            ? view({ characters: [view().characters[0]], merged: { ch_02: "ch_01" } })
            : view({ characters: [view().characters[0]], ignored: ["ch_02"] }),
        );
      },
      "PATCH /api/assets/ast_1/characters/ch_01": async (req) => {
        bodies.push(await req.json());
        return json(view());
      },
    });
    renderWithProviders(<App />, "/library/ast_1");
    const card = await screen.findByRole("listitem", { name: "未命名" });
    await userEvent.selectOptions(within(card).getByRole("combobox", { name: "把未命名合并到" }), "ch_01");
    const hidden = await screen.findByLabelText("已隐藏的人物");
    expect(hidden).toHaveTextContent("ch_02 已并入 ch_01");
    expect(screen.queryByRole("listitem", { name: "未命名" })).not.toBeInTheDocument();
    expect(bodies[0]).toEqual({ reset: false, merged_into: "ch_01" });

    await userEvent.click(within(hidden).getByRole("button", { name: "拆开" }));
    await waitFor(() => expect(bodies[1]).toEqual({ reset: true }));
  });

  it("shows an ignored character with a way back", async () => {
    const bodies: unknown[] = [];
    stubApi({
      ...base,
      "GET /api/assets/ast_1/index/characters": () =>
        json(view({ characters: [view().characters[0]], ignored: ["ch_02"], unmatched_edits: 1 })),
      "PATCH /api/assets/ast_1/characters/ch_02": async (req) => {
        bodies.push(await req.json());
        return json(view());
      },
    });
    renderWithProviders(<App />, "/library/ast_1");
    const hidden = await screen.findByLabelText("已隐藏的人物");
    expect(hidden).toHaveTextContent("已忽略 ch_02");
    expect(hidden).toHaveTextContent("有 1 条修改找不到对应的人物");
    await userEvent.click(within(hidden).getByRole("button", { name: "恢复" }));
    await screen.findByRole("listitem", { name: "未命名" });
    expect(bodies).toEqual([{ reset: false, ignored: false }]);
  });

  it("shows a failed edit", async () => {
    stubApi({
      ...base,
      "GET /api/assets/ast_1/index/characters": () => json(view()),
      "PATCH /api/assets/ast_1/characters/ch_01": () =>
        json({ error: { code: "invalid_input", message: "cannot merge ch_01 into ch_01" } }, 422),
    });
    renderWithProviders(<App />, "/library/ast_1");
    const card = await screen.findByRole("listitem", { name: "Sintel" });
    await userEvent.click(within(card).getByRole("button", { name: "忽略" }));
    expect(await screen.findByText(/cannot merge ch_01 into ch_01/)).toBeInTheDocument();
  });
});
