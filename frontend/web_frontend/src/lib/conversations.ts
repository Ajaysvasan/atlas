import { useSyncExternalStore } from "react";
import { ApiError, api, describeError, type Exchange, type QueryResponse } from "./api";

/**
 * The server is the record of which conversations and turns exist (GET /conversations and
 * /conversations/{id}/exchanges). The browser caches only what the server does not keep: each
 * answer's status, timings, retries and warnings, per user, in localStorage.
 */

export type ResponseMeta = Omit<QueryResponse, "answer">;

export interface Turn {
  id: string;
  query: string;
  askedAt: number;
  answer?: string;
  answeredAt?: number;
  /** Present for answers given in this browser; the server does not keep it. */
  response?: ResponseMeta;
  /** A request that failed; shown for this session only. */
  error?: { code: string; message: string; requestId?: string };
}

export interface ConversationMeta {
  id: string;
  name: string;
  createdAt: number;
  lastActivityAt: number;
  exchangeCount: number;
}

export interface ThreadState {
  status: "loading" | "ready" | "missing" | "error";
  turns: Turn[];
  error?: string;
}

export interface ListState {
  status: "idle" | "loading" | "ready" | "error";
  items: ConversationMeta[];
  nextCursor: string | null;
  loadingMore: boolean;
  error?: string;
}

export interface PendingQuery {
  /** Null while the pipeline has not yet named a new conversation. */
  conversationId: string | null;
  query: string;
  startedAt: number;
  controller: AbortController;
}

interface State {
  userId: string | null;
  list: ListState;
  threads: Record<string, ThreadState>;
  pending: PendingQuery | null;
  /** A failed first question has no conversation to live in; it is kept here until dismissed. */
  orphanError: Turn | null;
}

export const PAGE_SIZE = 50;
const EXCHANGE_PAGE_SIZE = 200;
/** A conversation longer than this many pages is shown up to here. */
const MAX_EXCHANGE_PAGES = 20;
const MAX_CACHED_CONVERSATIONS = 300;
const MAX_CACHED_TURNS = 200;

const EMPTY_LIST: ListState = { status: "idle", items: [], nextCursor: null, loadingMore: false };

let state: State = { userId: null, list: EMPTY_LIST, threads: {}, pending: null, orphanError: null };
const listeners = new Set<() => void>();
/** Failed requests per conversation, kept for the session so a reload of the thread keeps them. */
let sessionErrors: Record<string, Turn[]> = {};

function emit(next: Partial<State>) {
  state = { ...state, ...next };
  listeners.forEach((l) => l());
}

function turnId(): string {
  return `t-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function time(iso: string | undefined): number | undefined {
  if (!iso) return undefined;
  const ms = Date.parse(iso);
  return Number.isNaN(ms) ? undefined : ms;
}

/* ------------------------------------------------------- response cache */

interface CachedAnswer {
  query: string;
  response: ResponseMeta;
}

type AnswerCache = Record<string, CachedAnswer[]>;

const cacheKey = (userId: string) => `atlas.answers.${userId}`;
const legacyKey = (userId: string) => `atlas.conversations.${userId}`;

function readCache(userId: string): AnswerCache {
  try {
    const raw = localStorage.getItem(cacheKey(userId));
    const parsed = raw ? (JSON.parse(raw) as AnswerCache) : {};
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function writeCache(userId: string, cache: AnswerCache) {
  const ids = Object.keys(cache);
  const kept: AnswerCache = {};
  // Object keys keep insertion order: the oldest conversations go first when over the cap.
  for (const id of ids.slice(Math.max(0, ids.length - MAX_CACHED_CONVERSATIONS))) {
    kept[id] = cache[id].slice(-MAX_CACHED_TURNS);
  }
  try {
    localStorage.setItem(cacheKey(userId), JSON.stringify(kept));
  } catch {
    /* quota or blocked storage: the metadata is a nicety */
  }
}

function remember(userId: string, conversationId: string, answer: CachedAnswer) {
  const cache = readCache(userId);
  const list = cache[conversationId] ?? [];
  delete cache[conversationId];
  cache[conversationId] = [...list, answer];
  writeCache(userId, cache);
}

/** The browser-only history of earlier versions becomes the response cache, once. */
function migrateLegacy(userId: string) {
  try {
    const raw = localStorage.getItem(legacyKey(userId));
    if (!raw) return;
    const old = JSON.parse(raw) as Array<{ id: string; turns: Array<{ query: string; response?: ResponseMeta }> }>;
    if (Array.isArray(old)) {
      const cache = readCache(userId);
      for (const c of old) {
        const answers = (c.turns ?? []).filter((t) => t.response).map((t) => ({ query: t.query, response: t.response! }));
        if (answers.length && !cache[c.id]) cache[c.id] = answers;
      }
      writeCache(userId, cache);
    }
    localStorage.removeItem(legacyKey(userId));
  } catch {
    /* unreadable legacy data is simply dropped */
  }
}

/** Server exchanges, with this browser's metadata attached to the answers it saw, matched in order by query. */
export function toTurns(exchanges: Exchange[], cached: CachedAnswer[] = []): Turn[] {
  const unused = [...cached];
  return exchanges.map((e) => {
    const turn: Turn = {
      id: e.exchangeId,
      query: e.query,
      askedAt: time(e.askedAt) ?? 0,
      answer: e.answer,
      answeredAt: time(e.answeredAt),
    };
    if (e.answer !== undefined) {
      const i = unused.findIndex((c) => c.query === e.query);
      if (i >= 0) turn.response = unused.splice(i, 1)[0].response;
    }
    return turn;
  });
}

function toMeta(c: {
  conversationId: string;
  conversationName: string;
  createdAt: string;
  lastActivityAt: string;
  exchangeCount: number;
}): ConversationMeta {
  const createdAt = time(c.createdAt) ?? 0;
  return {
    id: c.conversationId,
    name: c.conversationName || "Untitled",
    createdAt,
    lastActivityAt: time(c.lastActivityAt) ?? createdAt,
    exchangeCount: c.exchangeCount,
  };
}

function setThread(id: string, thread: ThreadState) {
  emit({ threads: { ...state.threads, [id]: thread } });
}

/** Moves a conversation to the top of the list with its new activity, adding it if it is new. */
function touch(meta: ConversationMeta) {
  const rest = state.list.items.filter((c) => c.id !== meta.id);
  emit({ list: { ...state.list, items: [meta, ...rest] } });
}

/* ------------------------------------------------------------------ store */

export const conversationStore = {
  subscribe(listener: () => void) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  getState: () => state,

  /** Switches to a user; called whenever the signed-in user changes. */
  activate(userId: string | null) {
    if (state.userId === userId) return;
    state.pending?.controller.abort();
    sessionErrors = {};
    if (userId) migrateLegacy(userId);
    emit({ userId, list: EMPTY_LIST, threads: {}, pending: null, orphanError: null });
  },

  get(id: string | undefined): ConversationMeta | undefined {
    return id ? state.list.items.find((c) => c.id === id) : undefined;
  },

  /** The first page of conversations, replacing what is shown. */
  async loadList(token: string) {
    const userId = state.userId;
    emit({ list: { ...state.list, status: state.list.items.length ? state.list.status : "loading", error: undefined } });
    try {
      const res = await api.listConversations(token, { limit: PAGE_SIZE });
      if (state.userId !== userId) return;
      emit({
        list: { status: "ready", items: res.conversations.map(toMeta), nextCursor: res.nextCursor, loadingMore: false },
      });
    } catch (err) {
      if (state.userId !== userId) return;
      emit({ list: { ...state.list, status: "error", error: describeError(err) } });
    }
  },

  async loadMore(token: string) {
    const { nextCursor, loadingMore } = state.list;
    if (!nextCursor || loadingMore) return;
    const userId = state.userId;
    emit({ list: { ...state.list, loadingMore: true } });
    try {
      const res = await api.listConversations(token, { limit: PAGE_SIZE, cursor: nextCursor });
      if (state.userId !== userId) return;
      const seen = new Set(state.list.items.map((c) => c.id));
      emit({
        list: {
          ...state.list,
          items: [...state.list.items, ...res.conversations.map(toMeta).filter((c) => !seen.has(c.id))],
          nextCursor: res.nextCursor,
          loadingMore: false,
        },
      });
    } catch (err) {
      if (state.userId !== userId) return;
      emit({ list: { ...state.list, loadingMore: false, error: describeError(err) } });
    }
  },

  /** Every exchange of one conversation (up to a cap), oldest first. */
  async loadThread(token: string, id: string) {
    const userId = state.userId;
    const current = state.threads[id];
    setThread(id, { status: current?.status === "ready" ? "ready" : "loading", turns: current?.turns ?? [] });
    try {
      const exchanges: Exchange[] = [];
      let cursor: string | null = null;
      let name: string | undefined;
      for (let page = 0; page < MAX_EXCHANGE_PAGES; page++) {
        const res = await api.listExchanges(token, id, { limit: EXCHANGE_PAGE_SIZE, cursor });
        name = res.conversationName;
        exchanges.push(...res.exchanges);
        cursor = res.nextCursor;
        if (!cursor) break;
      }
      if (state.userId !== userId || !userId) return;
      const turns = [...toTurns(exchanges, readCache(userId)[id]), ...(sessionErrors[id] ?? [])].sort(
        (a, b) => a.askedAt - b.askedAt,
      );
      setThread(id, { status: "ready", turns });
      // A conversation opened by link before (or beyond) the listed page still gets a name.
      if (name && !conversationStore.get(id)) {
        const last = exchanges[exchanges.length - 1];
        const lastActivity = time(last?.answeredAt) ?? time(last?.askedAt) ?? Date.now();
        emit({
          list: {
            ...state.list,
            items: [
              ...state.list.items,
              { id, name, createdAt: time(exchanges[0]?.askedAt) ?? lastActivity, lastActivityAt: lastActivity, exchangeCount: exchanges.length },
            ].sort((a, b) => b.lastActivityAt - a.lastActivityAt),
          },
        });
      }
    } catch (err) {
      if (state.userId !== userId) return;
      if (err instanceof ApiError && err.code === "CONVERSATION_NOT_FOUND") {
        setThread(id, { status: "missing", turns: [] });
      } else {
        setThread(id, { status: "error", turns: state.threads[id]?.turns ?? [], error: describeError(err) });
      }
    }
  },

  dismissOrphanError() {
    emit({ orphanError: null });
  },

  cancelPending() {
    state.pending?.controller.abort();
  },

  /**
   * Sends one query. Resolves with the conversation id it landed in (new or existing), or null if
   * a first question failed. Only one query runs at a time: the pipeline itself is single-occupancy.
   */
  async ask(token: string, conversationId: string | null, query: string): Promise<string | null> {
    if (state.pending) throw new Error("A query is already running.");
    const userId = state.userId;
    const controller = new AbortController();
    const askedAt = Date.now();
    emit({ pending: { conversationId, query, startedAt: askedAt, controller }, orphanError: null });

    try {
      const res = await api.query(token, { conversationId, query }, controller.signal);
      if (state.userId !== userId || !userId) return null;
      const { answer, ...response } = res;
      const now = Date.now();
      const turn: Turn = { id: turnId(), query, askedAt, answer, answeredAt: now, response };
      remember(userId, res.conversationId, { query, response });

      const existing = conversationStore.get(res.conversationId);
      touch({
        id: res.conversationId,
        name: res.conversationName || existing?.name || "Untitled",
        createdAt: existing?.createdAt ?? askedAt,
        lastActivityAt: now,
        exchangeCount: (existing?.exchangeCount ?? 0) + 1,
      });
      const thread = state.threads[res.conversationId];
      if (thread?.status === "ready" || res.newConversation) {
        setThread(res.conversationId, { status: "ready", turns: [...(thread?.turns ?? []), turn] });
      }
      emit({ pending: null });
      return res.conversationId;
    } catch (err) {
      const apiErr = err instanceof ApiError ? err : null;
      const turn: Turn = {
        id: turnId(),
        query,
        askedAt,
        error: { code: apiErr?.code ?? "ERROR", message: describeError(err), requestId: apiErr?.requestId },
      };
      if (state.userId !== userId) return null;
      if (conversationId) {
        sessionErrors[conversationId] = [...(sessionErrors[conversationId] ?? []), turn];
        const thread = state.threads[conversationId];
        setThread(conversationId, { status: thread?.status ?? "ready", turns: [...(thread?.turns ?? []), turn] });
        emit({ pending: null });
        return conversationId;
      }
      emit({ pending: null, orphanError: turn });
      return null;
    }
  },
};

export function useConversations(): State {
  return useSyncExternalStore(conversationStore.subscribe, conversationStore.getState, conversationStore.getState);
}

/* ------------------------------------------------------- uploaded files */

/** What this browser has uploaded; the API has no listing endpoint for documents. */
export interface UploadedDocument {
  documentId: string;
  filename: string;
  chunks?: number;
  storedBytes?: number;
  uploadedAt: number;
}

const docsKey = (userId: string) => `atlas.documents.${userId}`;

export function loadDocuments(userId: string): UploadedDocument[] {
  try {
    const raw = localStorage.getItem(docsKey(userId));
    const parsed = raw ? (JSON.parse(raw) as UploadedDocument[]) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveDocuments(userId: string, docs: UploadedDocument[]): void {
  try {
    localStorage.setItem(docsKey(userId), JSON.stringify(docs.slice(0, 500)));
  } catch {
    /* ignore */
  }
}
