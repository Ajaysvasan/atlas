import { useEffect, useRef, useState, type DragEvent } from "react";
import { SectionLabel } from "../components/SectionLabel";
import { api, describeError, fileExtension, formatBytes, UPLOAD_LIMITS, type IngestFileResult, type IngestResponse } from "../lib/api";
import { useSession } from "../lib/auth";
import { loadDocuments, saveDocuments, type UploadedDocument } from "../lib/conversations";
import { formatDuration, formatStamp } from "../lib/format";

interface Staged {
  key: string;
  file: File;
  problem?: string;
}

/** The same checks the server makes first; the server still checks content, not just names. */
export function checkFile(file: File): string | undefined {
  const ext = fileExtension(file.name);
  if (!(UPLOAD_LIMITS.extensions as readonly string[]).includes(ext))
    return ext ? `.${ext} files are not accepted.` : "Files need an extension.";
  if (file.size === 0) return "The file is empty.";
  if (file.size > UPLOAD_LIMITS.maxFileBytes) return `Over the ${formatBytes(UPLOAD_LIMITS.maxFileBytes)} limit.`;
  return undefined;
}

const STATUS_LABEL: Record<IngestFileResult["status"], string> = {
  INGESTED: "Ingested",
  REJECTED: "Rejected",
  FAILED: "Failed",
  NOT_INGESTED: "Not ingested",
};

const HTTP_SUMMARY: Record<number, string> = {
  200: "Every file was ingested.",
  207: "Some files were ingested.",
  413: "Every file was too large, or the request was over the limit.",
  415: "Every file was an unsupported type.",
  422: "Nothing could be ingested.",
  507: "The ingestion storage quota is exhausted.",
};

type Phase = { kind: "idle" } | { kind: "uploading"; progress: number } | { kind: "ingesting"; since: number };

export function Documents() {
  const session = useSession();
  const [staged, setStaged] = useState<Staged[]>([]);
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const [result, setResult] = useState<{ httpStatus: number; body: IngestResponse } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [history, setHistory] = useState<UploadedDocument[]>(() => loadDocuments(session.userId));
  const input = useRef<HTMLInputElement>(null);
  const controller = useRef<AbortController | null>(null);

  const valid = staged.filter((s) => !s.problem);
  const totalBytes = valid.reduce((n, s) => n + s.file.size, 0);
  const requestProblem =
    valid.length > UPLOAD_LIMITS.maxFiles
      ? `At most ${UPLOAD_LIMITS.maxFiles} files per upload.`
      : totalBytes > UPLOAD_LIMITS.maxRequestBytes
        ? `At most ${formatBytes(UPLOAD_LIMITS.maxRequestBytes)} per upload.`
        : null;
  const busy = phase.kind !== "idle";
  const canUpload = !busy && valid.length > 0 && !requestProblem;

  function add(files: FileList | File[]) {
    const incoming = Array.from(files).map((file) => ({
      key: `${file.name}-${file.size}-${file.lastModified}`,
      file,
      problem: checkFile(file),
    }));
    setStaged((prev) => {
      const seen = new Set(prev.map((s) => s.key));
      return [...prev, ...incoming.filter((s) => !seen.has(s.key))];
    });
    setResult(null);
    setError(null);
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    if (!busy && e.dataTransfer.files.length) add(e.dataTransfer.files);
  }

  async function upload() {
    if (!canUpload) return;
    controller.current = new AbortController();
    setPhase({ kind: "uploading", progress: 0 });
    setError(null);
    setResult(null);
    try {
      const res = await api.ingest(
        session.accessToken,
        valid.map((s) => s.file),
        {
          signal: controller.current.signal,
          onUploadProgress: (p) =>
            setPhase((cur) => (p >= 1 ? { kind: "ingesting", since: Date.now() } : cur.kind === "uploading" ? { kind: "uploading", progress: p } : cur)),
        },
      );
      setResult(res);
      const now = Date.now();
      const added: UploadedDocument[] = res.body.files
        .filter((f) => f.status === "INGESTED" && f.documentId)
        .map((f) => ({ documentId: f.documentId!, filename: f.filename, chunks: f.chunks, storedBytes: f.storedBytes, uploadedAt: now }));
      if (added.length) {
        const next = [...added, ...history.filter((h) => !added.some((a) => a.documentId === h.documentId))];
        setHistory(next);
        saveDocuments(session.userId, next);
      }
      setStaged([]);
    } catch (err) {
      setError(describeError(err));
    } finally {
      setPhase({ kind: "idle" });
      controller.current = null;
    }
  }

  return (
    <div className="px-gutter pb-24 pt-12">
      <div className="grid grid-cols-12 gap-x-gutter">
        <SectionLabel index="02" en="Cargo" ja="積荷" className="col-span-12 mb-10 lg:col-span-10 lg:col-start-2" />
        <h1 className="display col-span-12 text-[clamp(3rem,8vw,7rem)] leading-[0.9] lg:col-span-7 lg:col-start-2">
          Give Atlas something to read.
        </h1>
        <p className="col-span-12 mt-6 max-w-xl text-ink-dim lg:col-span-4 lg:col-start-9 lg:mt-3">
          Uploads are private: only your own questions can retrieve them. Up to {UPLOAD_LIMITS.maxFiles} files and{" "}
          {formatBytes(UPLOAD_LIMITS.maxRequestBytes)} at a time, {formatBytes(UPLOAD_LIMITS.maxFileBytes)} each.
        </p>

        {/* drop zone */}
        <div className="col-span-12 mt-14 lg:col-span-10 lg:col-start-2">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              if (!busy) setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            className={`relative flex min-h-56 flex-col items-start justify-end gap-4 border p-6 transition-colors duration-500 sm:p-10 ${
              dragging ? "border-accent bg-accent-dim/30" : "border-dashed border-rule-strong"
            }`}
          >
            <span aria-hidden="true" className={`absolute inset-y-0 left-0 w-px ${dragging ? "bg-accent" : "bg-transparent"}`} />
            <p className="label">
              {UPLOAD_LIMITS.extensions.map((e) => `.${e}`).join("  ")}
            </p>
            <p className="display text-3xl sm:text-4xl">Drop files here</p>
            <button type="button" className="btn" onClick={() => input.current?.click()} disabled={busy}>
              Choose files
            </button>
            <input
              ref={input}
              type="file"
              multiple
              className="sr-only"
              tabIndex={-1}
              accept={UPLOAD_LIMITS.extensions.map((e) => `.${e}`).join(",")}
              onChange={(e) => {
                if (e.target.files) add(e.target.files);
                e.target.value = "";
              }}
            />
          </div>

          {/* staged files */}
          {staged.length > 0 && (
            <div className="mt-10">
              <p className="label mb-3">
                Selected <span className="text-ink-faint">/</span> <span lang="ja">選択中</span>
              </p>
              <ul className="border-t border-rule">
                {staged.map((s) => (
                  <li key={s.key} className="grid grid-cols-[minmax(0,1fr)_auto_auto] items-baseline gap-6 border-b border-rule py-3">
                    <span className="min-w-0">
                      <span className="block truncate">{s.file.name}</span>
                      {s.problem && <span className="text-sm text-danger">{s.problem}</span>}
                    </span>
                    <span className="label">{formatBytes(s.file.size)}</span>
                    <button
                      type="button"
                      className="label link"
                      disabled={busy}
                      onClick={() => setStaged((prev) => prev.filter((p) => p.key !== s.key))}
                      aria-label={`Remove ${s.file.name}`}
                    >
                      Remove
                    </button>
                  </li>
                ))}
              </ul>
              <div className="mt-6 flex flex-wrap items-center gap-6">
                <button type="button" className="btn btn-solid" disabled={!canUpload} onClick={upload}>
                  Ingest {valid.length} {valid.length === 1 ? "file" : "files"} · {formatBytes(totalBytes)}
                  <span aria-hidden="true">→</span>
                </button>
                {busy && (
                  <button type="button" className="label link" onClick={() => controller.current?.abort()}>
                    Cancel
                  </button>
                )}
                {requestProblem && <p className="text-sm text-danger">{requestProblem}</p>}
              </div>
            </div>
          )}

          {busy && <Progress phase={phase} />}

          {error && (
            <p role="alert" className="mt-8 border-l border-danger py-1 pl-4">
              {error}
            </p>
          )}

          {result && <IngestReport result={result} />}
        </div>

        {/* what this browser has uploaded */}
        <div className="col-span-12 mt-24 lg:col-span-10 lg:col-start-2">
          <p className="label mb-3">
            Uploaded from this browser <span className="text-ink-faint">/</span> <span lang="ja">取り込み済み</span>
          </p>
          {history.length === 0 ? (
            <p className="border-t border-rule pt-4 text-sm text-ink-faint">Nothing yet.</p>
          ) : (
            <ul className="border-t border-rule">
              {history.map((d) => (
                <li
                  key={d.documentId}
                  className="grid grid-cols-[minmax(0,1fr)_auto] items-baseline gap-x-6 border-b border-rule py-3 sm:grid-cols-[minmax(0,1fr)_8rem_10rem]"
                >
                  <span className="truncate" title={`${d.filename} — ${d.documentId}`}>
                    {d.filename}
                  </span>
                  <span className="label whitespace-nowrap">{d.chunks ?? "—"} chunks</span>
                  <span className="label hidden whitespace-nowrap text-right sm:block">{formatStamp(d.uploadedAt)}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}

function Progress({ phase }: { phase: Phase }) {
  const pct = phase.kind === "uploading" ? Math.round(phase.progress * 100) : 100;
  return (
    <div className="mt-10" role="status" aria-live="polite">
      <div className="flex justify-between">
        <p className="label text-accent-ink">{phase.kind === "uploading" ? `Uploading · ${pct}%` : "Ingesting · parsing, chunking, embedding"}</p>
        {phase.kind === "ingesting" && <Elapsed since={phase.since} />}
      </div>
      <div className="mt-3 h-px w-full bg-rule">
        <div
          className={`h-px bg-accent transition-[width] duration-300 ${phase.kind === "ingesting" ? "animate-pulse" : ""}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function Elapsed({ since }: { since: number }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);
  return <p className="label">{formatDuration(Math.max(0, now - since), true)}</p>;
}

function IngestReport({ result }: { result: { httpStatus: number; body: IngestResponse } }) {
  const { body, httpStatus } = result;
  const used = body.storageUsedBytes;
  const limit = body.storageLimitBytes;
  const share = used !== undefined && limit ? Math.min(1, used / limit) : null;

  return (
    <section className="mt-12" aria-label="Ingestion report">
      <div className="flex flex-wrap items-baseline justify-between gap-4 border-b border-rule pb-4">
        <p className="display text-3xl">
          {HTTP_SUMMARY[httpStatus] ?? body.status}
        </p>
        <p className="label">
          {body.succeeded} ingested · {body.failed} not · {formatDuration(body.executionTimeMs)}
        </p>
      </div>
      <ul>
        {body.files.map((f) => (
          <li key={f.index} className="grid grid-cols-[2.5rem_minmax(0,1fr)] gap-x-4 border-b border-rule py-4 sm:grid-cols-[2.5rem_minmax(0,1fr)_9rem_12rem]">
            <span className="font-mono text-xs text-accent-ink">{String(f.index + 1).padStart(2, "0")}</span>
            <span className="min-w-0">
              <span className="block truncate">{f.filename}</span>
              {f.message && <span className="text-sm text-ink-dim">{f.message}</span>}
            </span>
            <span className={`label col-start-2 sm:col-start-auto ${f.status === "INGESTED" ? "text-ink" : "text-danger"}`}>
              {STATUS_LABEL[f.status] ?? f.status}
              {f.errorCode && <span className="block text-[10px] text-ink-faint">{f.errorCode}</span>}
            </span>
            <span className="label col-start-2 sm:col-start-auto">
              {f.status === "INGESTED" ? `${f.chunks ?? 0} chunks · ${formatBytes(f.storedBytes)}` : ""}
            </span>
          </li>
        ))}
      </ul>
      {share !== null && (
        <div className="mt-8 max-w-md">
          <div className="flex justify-between">
            <p className="label">Storage quota</p>
            <p className="label">
              {formatBytes(used)} / {formatBytes(limit)}
            </p>
          </div>
          <div className="mt-2 h-px bg-rule">
            <div className="h-px bg-accent" style={{ width: `${share * 100}%` }} />
          </div>
          <p className="mt-2 text-xs text-ink-faint">One pool, shared by every account.</p>
        </div>
      )}
    </section>
  );
}
