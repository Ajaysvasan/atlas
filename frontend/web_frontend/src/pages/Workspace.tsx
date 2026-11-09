import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode, type SubmitEvent } from "react";
import { createPortal } from "react-dom";
import { Link, NavLink, useNavigate, useParams } from "react-router";
import { Answer } from "../components/Answer";
import { Meridian, MeridianTick } from "../components/Meridian";
import { QUERY_MAX_CHARS } from "../lib/api";
import { useSession } from "../lib/auth";
import {
  conversationStore,
  useConversations,
  type ConversationMeta,
  type ListState,
  type ThreadState,
  type Turn,
} from "../lib/conversations";
import { formatDuration, formatStamp } from "../lib/format";

const EXAMPLES = [
  "Explain how PostgreSQL MVCC decides which row version a transaction sees.",
  "When should SQLite run in WAL mode, and what does it cost?",
  "How does Cassandra compaction choose which SSTables to merge?",
];

export function Workspace() {
  const { conversationId } = useParams();
  const session = useSession();
  const { list, threads, pending, orphanError } = useConversations();
  const conversation = conversationStore.get(conversationId);
  const thread = conversationId ? threads[conversationId] : undefined;
  const [railOpen, setRailOpen] = useState(false);

  useEffect(() => {
    if (list.status === "idle") void conversationStore.loadList(session.accessToken);
  }, [list.status, session.accessToken]);

  // Opening a conversation reads its turns from the server, unless they are already here.
  useEffect(() => {
    if (conversationId && !conversationStore.getState().threads[conversationId]) {
      void conversationStore.loadThread(session.accessToken, conversationId);
    }
  }, [conversationId, session.accessToken]);

  const missing = Boolean(conversationId && thread?.status === "missing");
  const title = conversation?.name ?? (conversationId ? (missing ? "Unknown conversation" : "Loading") : "Untitled");

  return (
    <div className="grid h-[calc(100svh-3.5rem)] grid-cols-1 lg:grid-cols-[19rem_minmax(0,1fr)]">
      <ConversationRail
        list={list}
        activeId={conversationId}
        pendingId={pending?.conversationId ?? null}
        className="hidden border-r border-rule lg:flex"
      />
      {railOpen &&
        createPortal(<RailDrawer list={list} activeId={conversationId} onClose={() => setRailOpen(false)} />, document.body)}

      <section className="flex min-h-0 flex-col" aria-label="Conversation">
        <div className="flex items-center justify-between gap-4 border-b border-rule px-gutter py-3">
          <div className="min-w-0">
            <p className="label">
              <span className="text-accent">{conversationId ? "Conversation" : "New"}</span>
              <span className="mx-2 text-ink-faint">/</span>
              <span lang="ja">会話</span>
            </p>
            <h1 className="truncate font-display text-lg font-bold tracking-[-0.01em]">{title}</h1>
          </div>
          <button type="button" className="label link shrink-0 lg:hidden" onClick={() => setRailOpen(true)}>
            History{list.status === "ready" ? ` (${list.items.length}${list.nextCursor ? "+" : ""})` : ""}
          </button>
        </div>

        {missing ? (
          <UnknownConversation />
        ) : conversationId ? (
          <Thread conversationId={conversationId} thread={thread} />
        ) : (
          <NewConversation orphanError={orphanError} />
        )}

        <Composer conversation={conversation} conversationId={conversationId} disabled={missing} />
      </section>
    </div>
  );
}

/* ------------------------------------------------------------------ rail */

function ConversationRail({
  list,
  activeId,
  pendingId,
  className = "",
  onNavigate,
}: {
  list: ListState;
  activeId?: string;
  pendingId?: string | null;
  className?: string;
  onNavigate?: () => void;
}) {
  const session = useSession();
  return (
    <aside className={`min-h-0 flex-col ${className}`} aria-label="Conversations">
      <div className="flex items-center justify-between px-gutter py-4 lg:px-6">
        <p className="label">
          History <span className="text-ink-faint">/</span> <span lang="ja">履歴</span>
        </p>
        <Link to="/workspace" onClick={onNavigate} className="label link text-ink">
          + New
        </Link>
      </div>
      <nav className="scrollbar-thin min-h-0 flex-1 overflow-y-auto border-t border-rule" aria-busy={list.status === "loading"}>
        {list.status === "loading" && <p className="label px-gutter py-6 lg:px-6">Loading</p>}
        {list.status === "error" && (
          <div className="px-gutter py-6 lg:px-6" role="alert">
            <p className="text-sm text-ink-dim">{list.error}</p>
            <button type="button" className="label link mt-3" onClick={() => void conversationStore.loadList(session.accessToken)}>
              Retry
            </button>
          </div>
        )}
        {list.status === "ready" && list.items.length === 0 && (
          <p className="px-gutter py-6 text-sm text-ink-faint lg:px-6">No conversations yet.</p>
        )}
        <ul>
          {list.items.map((c) => (
            <RailItem key={c.id} c={c} active={c.id === activeId} running={pendingId === c.id} onNavigate={onNavigate} />
          ))}
        </ul>
        {list.nextCursor && (
          <div className="px-gutter py-4 lg:px-6">
            <button
              type="button"
              className="label link"
              disabled={list.loadingMore}
              onClick={() => void conversationStore.loadMore(session.accessToken)}
            >
              {list.loadingMore ? "Loading" : "Load more"}
            </button>
          </div>
        )}
      </nav>
    </aside>
  );
}

function RailItem({ c, active, running, onNavigate }: { c: ConversationMeta; active: boolean; running: boolean; onNavigate?: () => void }) {
  return (
    <li className="relative border-b border-rule">
      <NavLink
        to={`/workspace/${encodeURIComponent(c.id)}`}
        onClick={onNavigate}
        className={({ isActive }) =>
          `block px-gutter py-4 transition-colors duration-500 lg:px-6 ${isActive ? "bg-base-2" : "hover:bg-base-2/60"}`
        }
      >
        {active && <span aria-hidden="true" className="absolute inset-y-0 left-0 w-px bg-accent" />}
        <span className="block truncate font-display text-[15px] font-bold tracking-[-0.01em]">{c.name}</span>
        <span className="label mt-1 block text-[10px]">
          {formatStamp(c.lastActivityAt)} · {c.exchangeCount} {c.exchangeCount === 1 ? "turn" : "turns"}
          {running && <span className="ml-2 text-accent">· Running</span>}
        </span>
      </NavLink>
    </li>
  );
}

function RailDrawer({ list, activeId, onClose }: { list: ListState; activeId?: string; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div role="dialog" aria-modal="true" aria-label="Conversation history" className="fixed inset-0 z-50 flex lg:hidden">
      <div className="flex w-[min(22rem,88vw)] flex-col border-r border-rule bg-base">
        <div className="flex h-14 items-center justify-end px-gutter">
          <button type="button" className="label" onClick={onClose} autoFocus>
            Close
          </button>
        </div>
        <ConversationRail list={list} activeId={activeId} className="flex min-h-0 flex-1" onNavigate={onClose} />
      </div>
      <button type="button" aria-label="Close history" className="flex-1 bg-base/70" onClick={onClose} />
    </div>
  );
}

/* ---------------------------------------------------------------- thread */

function UnknownConversation() {
  return (
    <div className="flex flex-1 flex-col justify-center px-gutter">
      <p className="label mb-4 text-accent">404</p>
      <p className="max-w-xl font-display text-3xl font-bold tracking-[-0.02em]">No such conversation.</p>
      <p className="mt-4 max-w-xl text-ink-dim">It does not exist, or it belongs to another account.</p>
      <Link to="/workspace" className="btn mt-8 self-start">
        Start a new conversation
      </Link>
    </div>
  );
}

function useScrollToEnd(deps: unknown[]) {
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end", behavior: "smooth" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return end;
}

function Timeline({ turns, running, children }: { turns: Turn[]; running: boolean; children?: ReactNode }) {
  return (
    <div className="relative mx-auto max-w-4xl px-gutter py-12">
      {/* the meridian as the conversation's timeline */}
      <Meridian running={running} className="absolute bottom-0 left-gutter top-0" />
      <ol className="space-y-16 pl-8 sm:pl-12">
        {turns.map((t, i) => (
          <TurnView key={t.id} turn={t} index={i + 1} />
        ))}
        {children}
      </ol>
    </div>
  );
}

function Thread({ conversationId, thread }: { conversationId: string; thread?: ThreadState }) {
  const session = useSession();
  const { pending } = useConversations();
  const showPending = pending?.conversationId === conversationId;
  const turns = thread?.turns ?? [];
  const end = useScrollToEnd([turns.length, showPending, thread?.status]);

  return (
    <div className="scrollbar-thin min-h-0 flex-1 overflow-y-auto" aria-live="polite" aria-busy={thread?.status === "loading"}>
      {(!thread || thread.status === "loading") && turns.length === 0 && (
        <p className="label mx-auto max-w-4xl px-gutter py-12">Loading the conversation</p>
      )}
      {thread?.status === "error" && (
        <div role="alert" className="mx-auto max-w-4xl px-gutter pt-8">
          <p className="border-l border-danger py-1 pl-4 text-sm">{thread.error}</p>
          <button
            type="button"
            className="label link ml-4 mt-3"
            onClick={() => void conversationStore.loadThread(session.accessToken, conversationId)}
          >
            Retry
          </button>
        </div>
      )}
      {(turns.length > 0 || showPending) && (
        <Timeline turns={turns} running={showPending}>
          {showPending && pending && <PendingTurn query={pending.query} startedAt={pending.startedAt} index={turns.length + 1} />}
        </Timeline>
      )}
      <div ref={end} />
    </div>
  );
}

function NewConversation({ orphanError }: { orphanError: Turn | null }) {
  const { pending } = useConversations();
  const showPending = Boolean(pending && pending.conversationId === null);
  const end = useScrollToEnd([showPending, orphanError]);
  if (!showPending && !orphanError) return <EmptyState />;
  return (
    <div className="scrollbar-thin min-h-0 flex-1 overflow-y-auto" aria-live="polite">
      <Timeline turns={[]} running={showPending}>
        {orphanError && <TurnView turn={orphanError} index={1} onDismiss={() => conversationStore.dismissOrphanError()} />}
        {showPending && pending && <PendingTurn query={pending.query} startedAt={pending.startedAt} index={1} />}
      </Timeline>
      <div ref={end} />
    </div>
  );
}

function TurnView({ turn, index, onDismiss }: { turn: Turn; index: number; onDismiss?: () => void }) {
  const r = turn.response;
  return (
    <li className="relative">
      <div className="-ml-8 mb-4 sm:-ml-12">
        <MeridianTick label={`T—${String(index).padStart(2, "0")} · ${formatStamp(turn.askedAt)}`} />
      </div>
      <p className="mb-6 whitespace-pre-wrap font-display text-[clamp(1.15rem,2vw,1.45rem)] font-bold leading-[1.3] tracking-[-0.015em]">
        {turn.query}
      </p>

      {turn.answer !== undefined && <Answer text={turn.answer} />}

      {turn.answer === undefined && !turn.error && (
        <p className="border-l border-rule-strong py-1 pl-4 text-sm text-ink-dim">No answer was recorded for this question.</p>
      )}

      {turn.error && (
        <div role="alert" className="border-l border-danger py-1 pl-4">
          <p className="label mb-1 text-danger">{turn.error.code}</p>
          <p className="text-ink">{turn.error.message}</p>
          {turn.error.requestId && <p className="label mt-2 normal-case tracking-[0.08em]">Request {turn.error.requestId}</p>}
          {onDismiss && (
            <button type="button" className="label link mt-3" onClick={onDismiss}>
              Dismiss
            </button>
          )}
        </div>
      )}

      {r && (r.warnings.length > 0 || !r.persisted) && (
        <ul className="mt-4 space-y-1 border-l border-accent pl-4 text-sm text-ink-dim">
          {r.warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
          {!r.persisted && r.warnings.length === 0 && <li>This turn was not saved to memory; the next question will not see it.</li>}
        </ul>
      )}

      {r ? (
        <dl className="mt-6 flex flex-wrap gap-x-6 gap-y-1 border-t border-rule pt-3">
          <Meta k="Status" v={r.status === "COMPLETED" ? "Completed" : "With warnings"} accent={r.status !== "COMPLETED"} />
          <Meta k="Time" v={formatDuration(r.executionTimeMs)} />
          <Meta k="Retries" v={String(r.retryCount)} />
          <Meta k="Request" v={r.requestId.slice(0, 18)} title={r.requestId} />
        </dl>
      ) : (
        turn.answeredAt !== undefined && (
          <dl className="mt-6 flex flex-wrap gap-x-6 gap-y-1 border-t border-rule pt-3">
            <Meta k="Answered" v={formatStamp(turn.answeredAt)} />
          </dl>
        )
      )}
    </li>
  );
}

function Meta({ k, v, accent = false, title }: { k: string; v: string; accent?: boolean; title?: string }) {
  return (
    <div className="flex gap-2" title={title}>
      <dt className="label text-[10px] text-ink-faint">{k}</dt>
      <dd className={`label text-[10px] ${accent ? "text-accent" : "text-ink-dim"}`}>{v}</dd>
    </div>
  );
}

function useElapsed(since: number): number {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);
  return Math.max(0, now - since);
}

function PendingTurn({ query, startedAt, index }: { query: string; startedAt: number; index: number }) {
  const elapsed = useElapsed(startedAt);
  return (
    <li className="relative">
      <div className="-ml-8 mb-4 sm:-ml-12">
        <MeridianTick label={`T—${String(index).padStart(2, "0")} · ${formatStamp(startedAt)}`} active />
      </div>
      <p className="mb-6 whitespace-pre-wrap font-display text-[clamp(1.15rem,2vw,1.45rem)] font-bold leading-[1.3] tracking-[-0.015em]">
        {query}
      </p>
      <div className="flex flex-wrap items-baseline gap-x-6 gap-y-2">
        <p className="label text-accent" role="status">
          Running · {formatDuration(elapsed, true)}
        </p>
        <p className="text-sm text-ink-dim">Decomposing, recalling, retrieving, planning and critiquing. On a CPU this takes minutes.</p>
      </div>
      <button
        type="button"
        className="label link mt-4"
        onClick={() => conversationStore.cancelPending()}
        title="Stops waiting in this browser. A query that has already started still finishes on the server."
      >
        Stop waiting
      </button>
    </li>
  );
}

function EmptyState() {
  return (
    <div className="scrollbar-thin flex min-h-0 flex-1 flex-col justify-center overflow-y-auto px-gutter py-12">
      <div className="mx-auto w-full max-w-4xl">
        <p className="label mb-6">
          <span className="text-accent">01</span>
          <span className="mx-3 text-ink-faint">—</span>
          Ask <span className="text-ink-faint">/</span> <span lang="ja">質問</span>
        </p>
        <h2 className="max-w-3xl font-display text-[clamp(2.4rem,6vw,5rem)] font-extrabold leading-[0.95] tracking-[-0.04em]">
          What do you want to know?
        </h2>
        <p className="mt-6 max-w-xl text-ink-dim">
          Atlas answers from the reference corpus and from the documents you upload. Follow-up questions keep the conversation’s
          memory.
        </p>
        <ul className="mt-12 max-w-3xl border-t border-rule">
          {EXAMPLES.map((q, i) => (
            <li key={q} className="border-b border-rule">
              <button
                type="button"
                className="group flex w-full items-baseline gap-6 py-4 text-left transition-transform duration-700 hover:translate-x-2"
                onClick={() => window.dispatchEvent(new CustomEvent("atlas:prefill", { detail: q }))}
              >
                <span className="font-mono text-xs text-accent">{String(i + 1).padStart(2, "0")}</span>
                <span className="flex-1 text-ink-dim transition-colors group-hover:text-ink">{q}</span>
                <span aria-hidden="true" className="text-accent opacity-0 transition-opacity duration-500 group-hover:opacity-100">
                  →
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------- composer */

function Composer({
  conversation,
  conversationId,
  disabled,
}: {
  conversation?: ConversationMeta;
  conversationId?: string;
  disabled: boolean;
}) {
  const session = useSession();
  const navigate = useNavigate();
  const { pending } = useConversations();
  const [text, setText] = useState("");
  const area = useRef<HTMLTextAreaElement>(null);
  const currentId = useRef<string | undefined>(conversationId);
  currentId.current = conversationId;

  useEffect(() => {
    const onPrefill = (e: Event) => {
      setText((e as CustomEvent<string>).detail);
      area.current?.focus();
    };
    window.addEventListener("atlas:prefill", onPrefill);
    return () => window.removeEventListener("atlas:prefill", onPrefill);
  }, []);

  useEffect(() => {
    const el = area.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 240)}px`;
  }, [text]);

  const busyElsewhere = pending && (pending.conversationId ?? null) !== (conversationId ?? null);
  const busyName = busyElsewhere
    ? pending?.conversationId
      ? conversationStore.get(pending.conversationId)?.name ?? "another conversation"
      : "a new conversation"
    : null;
  const trimmed = text.trim();
  const over = text.length > QUERY_MAX_CHARS;
  const canSend = !disabled && !pending && trimmed.length > 0 && !over;

  async function send(e?: SubmitEvent) {
    e?.preventDefault();
    if (!canSend) return;
    const askedFrom = conversationId ?? null;
    setText("");
    const landed = await conversationStore.ask(session.accessToken, askedFrom, trimmed);
    // A new conversation gets its own URL, unless the user has moved on meanwhile.
    if (landed && askedFrom === null && currentId.current === undefined && window.location.pathname === "/workspace") {
      navigate(`/workspace/${encodeURIComponent(landed)}`, { replace: true });
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void send();
    }
  }

  return (
    <form onSubmit={send} className="border-t border-rule bg-base px-gutter pb-4 pt-3">
      <div className="mx-auto max-w-4xl">
        {busyName && <p className="label mb-2 text-accent">Atlas is answering in {busyName}. One question runs at a time.</p>}
        <div className="flex items-end gap-4">
          <label htmlFor="query" className="sr-only">
            Your question
          </label>
          <textarea
            id="query"
            ref={area}
            rows={1}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKeyDown}
            disabled={disabled}
            placeholder={conversation || conversationId ? "Ask a follow-up" : "Ask a question"}
            className="field scrollbar-thin max-h-60 min-h-[2.75rem] flex-1 resize-none leading-relaxed placeholder:text-ink-faint"
            aria-invalid={over || undefined}
            aria-describedby="query-meta"
          />
          <button type="submit" className="btn btn-solid shrink-0" disabled={!canSend}>
            {pending && !busyElsewhere ? "Running" : "Ask"}
            <span aria-hidden="true">→</span>
          </button>
        </div>
        <div id="query-meta" className="mt-2 flex justify-end gap-4 sm:justify-between">
          <p className="label hidden text-[10px] text-ink-faint sm:block">Enter to send · Shift+Enter for a new line</p>
          <p className={`label text-[10px] ${over ? "text-danger" : "text-ink-faint"}`}>
            {text.length.toLocaleString()} / {QUERY_MAX_CHARS.toLocaleString()}
          </p>
        </div>
      </div>
    </form>
  );
}
