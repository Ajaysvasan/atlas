import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { conversationStore } from "../src/lib/conversations";

const ok = (body: object) => new Response(JSON.stringify(body), { status: 200 });
const answer = (id: string, name: string) => ({
  requestId: "r", conversationId: id, conversationName: name, newConversation: true, answer: "A",
  status: "COMPLETED", executionTimeMs: 1, retryCount: 0, warnings: [], persisted: true,
});

beforeEach(() => {
  localStorage.clear();
  conversationStore.activate(null);
  conversationStore.activate("user-1");
});
afterEach(() => vi.unstubAllGlobals());

describe("conversationStore", () => {
  it("records a new conversation and persists it per user", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(ok(answer("c1", "MVCC"))));
    const id = await conversationStore.ask("tok", null, "What is MVCC?");
    expect(id).toBe("c1");
    const state = conversationStore.getState();
    expect(state.pending).toBeNull();
    expect(state.conversations[0]).toMatchObject({ id: "c1", name: "MVCC" });
    expect(state.conversations[0].turns[0]).toMatchObject({ query: "What is MVCC?", answer: "A" });
    expect(JSON.parse(localStorage.getItem("atlas.conversations.user-1")!)).toHaveLength(1);
  });

  it("keeps each user's history apart", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(ok(answer("c1", "MVCC"))));
    await conversationStore.ask("tok", null, "q");
    conversationStore.activate("user-2");
    expect(conversationStore.getState().conversations).toEqual([]);
    conversationStore.activate("user-1");
    expect(conversationStore.getState().conversations).toHaveLength(1);
  });

  it("appends a failed follow-up to its conversation", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValueOnce(ok(answer("c1", "MVCC"))).mockResolvedValueOnce(
      new Response(JSON.stringify({ status: 404, code: "CONVERSATION_NOT_FOUND" }), { status: 404 }),
    ));
    await conversationStore.ask("tok", null, "first");
    await conversationStore.ask("tok", "c1", "second");
    const turns = conversationStore.get("c1")!.turns;
    expect(turns).toHaveLength(2);
    expect(turns[1].error?.code).toBe("CONVERSATION_NOT_FOUND");
  });

  it("keeps a failed first question as an orphan error, not a conversation", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ status: 503, code: "BACKEND_UNAVAILABLE" }), { status: 503 }),
    ));
    expect(await conversationStore.ask("tok", null, "q")).toBeNull();
    expect(conversationStore.getState().conversations).toEqual([]);
    expect(conversationStore.getState().orphanError?.error?.code).toBe("BACKEND_UNAVAILABLE");
  });

  it("runs one query at a time", async () => {
    let release!: (r: Response) => void;
    vi.stubGlobal("fetch", vi.fn().mockReturnValue(new Promise<Response>((r) => (release = r))));
    const first = conversationStore.ask("tok", null, "q1");
    await expect(conversationStore.ask("tok", null, "q2")).rejects.toThrow(/already running/);
    release(ok(answer("c1", "x")));
    await first;
  });

  it("can stop waiting for a running query", async () => {
    vi.stubGlobal("fetch", vi.fn((_: string, init: RequestInit) =>
      new Promise<Response>((_r, reject) => init.signal!.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))),
    ));
    const p = conversationStore.ask("tok", null, "q");
    conversationStore.cancelPending();
    expect(await p).toBeNull();
    expect(conversationStore.getState().orphanError?.error?.code).toBe("ABORTED");
  });
});
