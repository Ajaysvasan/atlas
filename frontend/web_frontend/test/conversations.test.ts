import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { conversationStore, toTurns } from "../src/lib/conversations";

const json = (status: number, body: unknown) => new Response(JSON.stringify(body), { status });
const answer = (id: string, name: string, extra: object = {}) => ({
  requestId: "r", conversationId: id, conversationName: name, newConversation: true, answer: "A",
  status: "COMPLETED", executionTimeMs: 1, retryCount: 0, warnings: [], persisted: true, ...extra,
});
const summary = (id: string, name: string, n = 1) => ({
  conversationId: id, conversationName: name, createdAt: "2026-11-09T10:00:00.000000+00:00",
  lastActivityAt: "2026-11-09T10:05:00.000000+00:00", exchangeCount: n,
});
const exchange = (id: string, query: string, answerText?: string) => ({
  exchangeId: id, query, askedAt: "2026-11-09T10:00:00.000000+00:00",
  ...(answerText === undefined ? {} : { answer: answerText, answeredAt: "2026-11-09T10:01:00.000000+00:00" }),
});

/** Routes fetch by path; each handler returns a Response. */
function server(routes: Record<string, (url: URL, init: RequestInit) => Response | Promise<Response>>) {
  const fetchMock = vi.fn(async (input: string, init: RequestInit) => {
    const url = new URL(input);
    const handler = Object.entries(routes).find(([path]) => url.pathname === path || url.pathname.startsWith(path + "/"));
    if (!handler) throw new Error(`unexpected ${url.pathname}`);
    return handler[1](url, init);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

beforeEach(() => {
  localStorage.clear();
  conversationStore.activate(null);
  conversationStore.activate("user-1");
});
afterEach(() => vi.unstubAllGlobals());

describe("the conversation list comes from the server", () => {
  it("loads the first page and follows the cursor for more", async () => {
    const fetchMock = server({
      "/conversations": (url) =>
        url.searchParams.get("cursor")
          ? json(200, { requestId: "r", conversations: [summary("c3", "Three")], nextCursor: null })
          : json(200, { requestId: "r", conversations: [summary("c1", "One"), summary("c2", "Two")], nextCursor: "abc" }),
    });
    await conversationStore.loadList("tok");
    expect(conversationStore.getState().list).toMatchObject({ status: "ready", nextCursor: "abc" });
    expect(conversationStore.getState().list.items.map((c) => c.id)).toEqual(["c1", "c2"]);
    expect(new URL(fetchMock.mock.calls[0][0]).searchParams.get("limit")).toBe("50");

    await conversationStore.loadMore("tok");
    expect(conversationStore.getState().list.items.map((c) => c.id)).toEqual(["c1", "c2", "c3"]);
    expect(conversationStore.getState().list.nextCursor).toBeNull();
    expect((fetchMock.mock.calls[1][1].headers as Record<string, string>).Authorization).toBe("Bearer tok");
  });

  it("reports a failed load", async () => {
    server({ "/conversations": () => json(503, { status: 503, code: "BACKEND_UNAVAILABLE" }) });
    await conversationStore.loadList("tok");
    expect(conversationStore.getState().list.status).toBe("error");
    expect(conversationStore.getState().list.error).toMatch(/pipeline/i);
  });
});

describe("threads", () => {
  it("are read from the server, pages joined, with cached answer metadata matched by query", async () => {
    localStorage.setItem("atlas.answers.user-1", JSON.stringify({
      c1: [{ query: "second", response: { requestId: "r2", status: "COMPLETED_WITH_WARNINGS", executionTimeMs: 9, retryCount: 1, warnings: ["w"], persisted: false } }],
    }));
    server({
      "/conversations/c1": (url) =>
        url.searchParams.get("cursor")
          ? json(200, { requestId: "r", conversationId: "c1", conversationName: "One", exchanges: [exchange("e3", "third")], nextCursor: null })
          : json(200, { requestId: "r", conversationId: "c1", conversationName: "One",
              exchanges: [exchange("e1", "first", "A1"), exchange("e2", "second", "A2")], nextCursor: "p2" }),
    });
    await conversationStore.loadThread("tok", "c1");
    const thread = conversationStore.getState().threads.c1;
    expect(thread.status).toBe("ready");
    expect(thread.turns.map((t) => [t.query, t.answer])).toEqual([["first", "A1"], ["second", "A2"], ["third", undefined]]);
    expect(thread.turns[0].response).toBeUndefined();
    expect(thread.turns[1].response?.requestId).toBe("r2");
    // Opened by link: it joins the list with the server's name.
    expect(conversationStore.get("c1")?.name).toBe("One");
  });

  it("marks another user's or a deleted conversation as missing", async () => {
    server({ "/conversations/nope": () => json(404, { status: 404, code: "CONVERSATION_NOT_FOUND" }) });
    await conversationStore.loadThread("tok", "nope");
    expect(conversationStore.getState().threads.nope.status).toBe("missing");
  });
});

describe("asking", () => {
  it("puts a new conversation at the top of the list and caches the answer's metadata", async () => {
    server({ "/query": () => json(200, answer("c9", "MVCC")) });
    const id = await conversationStore.ask("tok", null, "What is MVCC?");
    expect(id).toBe("c9");
    const state = conversationStore.getState();
    expect(state.pending).toBeNull();
    expect(state.list.items[0]).toMatchObject({ id: "c9", name: "MVCC", exchangeCount: 1 });
    expect(state.threads.c9.turns[0]).toMatchObject({ query: "What is MVCC?", answer: "A" });
    expect(JSON.parse(localStorage.getItem("atlas.answers.user-1")!).c9[0].query).toBe("What is MVCC?");
    expect(JSON.stringify(JSON.parse(localStorage.getItem("atlas.answers.user-1")!))).not.toContain('"answer"');
  });

  it("appends a failed follow-up to its thread for the session", async () => {
    server({
      "/conversations/c1": () => json(200, { requestId: "r", conversationId: "c1", conversationName: "One", exchanges: [exchange("e1", "first", "A1")], nextCursor: null }),
      "/query": () => json(503, { status: 503, code: "BACKEND_BUSY" }),
    });
    await conversationStore.loadThread("tok", "c1");
    await conversationStore.ask("tok", "c1", "second");
    expect(conversationStore.getState().threads.c1.turns.map((t) => t.error?.code ?? "ok")).toEqual(["ok", "BACKEND_BUSY"]);
    await conversationStore.loadThread("tok", "c1");
    expect(conversationStore.getState().threads.c1.turns).toHaveLength(2);
  });

  it("keeps a failed first question as an orphan error, not a conversation", async () => {
    server({ "/query": () => json(503, { status: 503, code: "BACKEND_UNAVAILABLE" }) });
    expect(await conversationStore.ask("tok", null, "q")).toBeNull();
    expect(conversationStore.getState().list.items).toEqual([]);
    expect(conversationStore.getState().orphanError?.error?.code).toBe("BACKEND_UNAVAILABLE");
  });

  it("runs one query at a time", async () => {
    let release!: (r: Response) => void;
    server({ "/query": () => new Promise<Response>((r) => (release = r)) });
    const first = conversationStore.ask("tok", null, "q1");
    await expect(conversationStore.ask("tok", null, "q2")).rejects.toThrow(/already running/);
    await vi.waitFor(() => expect(release).toBeDefined());
    release(json(200, answer("c1", "x")));
    await first;
  });

  it("can stop waiting for a running query", async () => {
    server({
      "/query": (_url, init) =>
        new Promise<Response>((_r, reject) => init.signal!.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))),
    });
    const p = conversationStore.ask("tok", null, "q");
    conversationStore.cancelPending();
    expect(await p).toBeNull();
    expect(conversationStore.getState().orphanError?.error?.code).toBe("ABORTED");
  });
});

describe("accounts", () => {
  it("keep their caches apart and the old browser-only history becomes metadata once", () => {
    localStorage.setItem("atlas.conversations.user-2", JSON.stringify([
      { id: "old", turns: [{ query: "q", response: { requestId: "r" } }, { query: "failed" }] },
    ]));
    conversationStore.activate("user-2");
    expect(localStorage.getItem("atlas.conversations.user-2")).toBeNull();
    expect(JSON.parse(localStorage.getItem("atlas.answers.user-2")!).old).toEqual([{ query: "q", response: { requestId: "r" } }]);
    expect(localStorage.getItem("atlas.answers.user-1")).toBeNull();
  });
});

describe("toTurns", () => {
  it("matches repeated questions to cached metadata in order", () => {
    const turns = toTurns(
      [exchange("e1", "same", "a"), exchange("e2", "same", "b")],
      [{ query: "same", response: { requestId: "first" } as never }, { query: "same", response: { requestId: "second" } as never }],
    );
    expect(turns.map((t) => t.response?.requestId)).toEqual(["first", "second"]);
  });
});
