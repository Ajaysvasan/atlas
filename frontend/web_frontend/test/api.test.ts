import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError, describeError, errorFromBody, formatBytes, newRequestId, onUnauthorized } from "../src/lib/api";

const json = (status: number, body: unknown, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });

afterEach(() => {
  vi.unstubAllGlobals();
  onUnauthorized(null);
});

describe("request ids", () => {
  it("match the API's 8-64 [A-Za-z0-9._-] rule", () => {
    for (let i = 0; i < 20; i++) expect(newRequestId()).toMatch(/^[A-Za-z0-9._-]{8,64}$/);
  });
});

describe("errorFromBody", () => {
  it("reads the API error shape and Retry-After", () => {
    const err = errorFromBody(
      503,
      { status: 503, code: "BACKEND_BUSY", message: "Atlas is busy; try again shortly.", requestId: "r-1" },
      new Headers({ "Retry-After": "7" }),
    );
    expect(err).toMatchObject({ status: 503, code: "BACKEND_BUSY", requestId: "r-1", retryAfter: 7 });
    expect(describeError(err)).toContain("7s");
  });

  it("exposes validation details per field", () => {
    const err = errorFromBody(400, { status: 400, code: "VALIDATION_FAILED", details: { username: "is required" } });
    expect(err.fieldErrors).toEqual({ username: "is required" });
  });

  it("falls back when the body is not JSON", () => {
    const err = errorFromBody(504, null);
    expect(err.code).toBe("HTTP_504");
    expect(err.message).toMatch(/too long/);
  });
});

describe("api.query", () => {
  it("sends the bearer token, a request id and the body, and returns the answer", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      json(200, { requestId: "r", conversationId: "c1", conversationName: "MVCC", newConversation: true, answer: "A", status: "COMPLETED", executionTimeMs: 5, retryCount: 0, warnings: [], persisted: true }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const res = await api.query("tok", { conversationId: null, query: "q" });
    expect(res.conversationId).toBe("c1");
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/query$/);
    expect(init.headers.Authorization).toBe("Bearer tok");
    expect(init.headers["X-Request-Id"]).toMatch(/^web-/);
    expect(JSON.parse(init.body)).toEqual({ conversationId: null, query: "q" });
  });

  it("calls the unauthorized handler on 401", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(401, { status: 401, code: "INVALID_TOKEN", message: "x" })));
    const handler = vi.fn();
    onUnauthorized(handler);
    await expect(api.query("tok", { conversationId: null, query: "q" })).rejects.toBeInstanceOf(ApiError);
    expect(handler).toHaveBeenCalledOnce();
  });

  it("does not sign out on a 401 from login (wrong password)", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json(401, { status: 401, code: "INVALID_CREDENTIALS" })));
    const handler = vi.fn();
    onUnauthorized(handler);
    await expect(api.login({ email: "a@b.c", password: "x" })).rejects.toMatchObject({ code: "INVALID_CREDENTIALS" });
    expect(handler).not.toHaveBeenCalled();
  });

  it("reports an unreachable API as NETWORK_ERROR", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(api.query("t", { conversationId: null, query: "q" })).rejects.toMatchObject({ code: "NETWORK_ERROR" });
  });
});

describe("formatBytes", () => {
  it("uses binary units", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(25 * 1024 * 1024)).toBe("25 MiB");
    expect(formatBytes(1 << 30)).toBe("1.0 GiB");
  });
});
