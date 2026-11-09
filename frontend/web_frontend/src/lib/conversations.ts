import { useSyncExternalStore } from "react";
import { ApiError, api, describeError, type QueryResponse } from "./api";

/**
 * The API has no endpoint that lists conversations or replays their turns, so the browser keeps
 * its own record, per user, in localStorage. The server stays the authority on ownership: a stale
 * id here just comes back 404 CONVERSATION_NOT_FOUND.
 */

export interface Turn {
  id: string;
  query: string;
  askedAt: number;
  answer?: string;
  response?: Omit<QueryResponse, "answer">;
  error?: { code: string; message: string; requestId?: string };
}

export interface Conversation {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  turns: Turn[];
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
  conversations: Conversation[];
  pending: PendingQuery | null;
  /** A failed first turn has no server conversation to live in; it is kept here until dismissed. */
  orphanError: Turn | null;
}

const MAX_CONVERSATIONS = 200;
const key = (userId: string) => `atlas.conversations.${userId}`;

let state: State = { userId: null, conversations: [], pending: null, orphanError: null };
const listeners = new Set<() => void>();

function emit(next: Partial<State>) {
  state = { ...state, ...next };
  listeners.forEach((l) => l());
}

function persist() {
  if (!state.userId) return;
  try {
    localStorage.setItem(key(state.userId), JSON.stringify(state.conversations.slice(0, MAX_CONVERSATIONS)));
  } catch {
    /* quota or blocked storage: history stays in memory for this page */
  }
}

function load(userId: string): Conversation[] {
  try {
    const raw = localStorage.getItem(key(userId));
    const parsed = raw ? (JSON.parse(raw) as Conversation[]) : [];
    return Array.isArray(parsed) ? parsed.filter((c) => c && typeof c.id === "string" && Array.isArray(c.turns)) : [];
  } catch {
    return [];
  }
}

function sorted(list: Conversation[]): Conversation[] {
  return [...list].sort((a, b) => b.updatedAt - a.updatedAt);
}

function turnId(): string {
  return `t-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export const conversationStore = {
  subscribe(listener: () => void) {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  getState: () => state,

  /** Switches to a user's history; called whenever the signed-in user changes. */
  activate(userId: string | null) {
    if (state.userId === userId) return;
    state.pending?.controller.abort();
    emit({ userId, conversations: userId ? sorted(load(userId)) : [], pending: null, orphanError: null });
  },

  get(id: string | undefined): Conversation | undefined {
    return id ? state.conversations.find((c) => c.id === id) : undefined;
  },

  remove(id: string) {
    emit({ conversations: state.conversations.filter((c) => c.id !== id) });
    persist();
  },

  clearAll() {
    if (state.userId) {
      try {
        localStorage.removeItem(key(state.userId));
      } catch {
        /* ignore */
      }
    }
    emit({ conversations: [], orphanError: null });
  },

  dismissOrphanError() {
    emit({ orphanError: null });
  },

  cancelPending() {
    state.pending?.controller.abort();
  },

  /**
   * Sends one query. Resolves with the conversation id it landed in (new or existing), or null if
   * a first turn failed. Only one query runs at a time: the pipeline itself is single-occupancy.
   */
  async ask(token: string, conversationId: string | null, query: string): Promise<string | null> {
    if (state.pending) throw new Error("A query is already running.");
    const controller = new AbortController();
    const askedAt = Date.now();
    emit({ pending: { conversationId, query, startedAt: askedAt, controller }, orphanError: null });

    try {
      const res = await api.query(token, { conversationId, query }, controller.signal);
      const { answer, ...meta } = res;
      const turn: Turn = { id: turnId(), query, askedAt, answer, response: meta };
      const existing = state.conversations.find((c) => c.id === res.conversationId);
      const now = Date.now();
      const conv: Conversation = existing
        ? { ...existing, name: res.conversationName || existing.name, updatedAt: now, turns: [...existing.turns, turn] }
        : { id: res.conversationId, name: res.conversationName || "Untitled", createdAt: now, updatedAt: now, turns: [turn] };
      emit({
        pending: null,
        conversations: sorted([conv, ...state.conversations.filter((c) => c.id !== conv.id)]),
      });
      persist();
      return conv.id;
    } catch (err) {
      const apiErr = err instanceof ApiError ? err : null;
      const turn: Turn = {
        id: turnId(),
        query,
        askedAt,
        error: { code: apiErr?.code ?? "ERROR", message: describeError(err), requestId: apiErr?.requestId },
      };
      const existing = conversationId ? state.conversations.find((c) => c.id === conversationId) : undefined;
      if (existing) {
        const conv = { ...existing, updatedAt: Date.now(), turns: [...existing.turns, turn] };
        emit({ pending: null, conversations: sorted([conv, ...state.conversations.filter((c) => c.id !== conv.id)]) });
        persist();
        return existing.id;
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

/** What this browser has uploaded; the API has no listing endpoint for documents either. */
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
