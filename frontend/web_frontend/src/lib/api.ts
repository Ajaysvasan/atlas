/**
 * REST client for the Java API (backend/atlas). Every contract here mirrors backend/BACKEND.md:
 * POST /register, POST /login, POST /query, POST /ingest and GET /actuator/health.
 */

export const API_BASE_URL: string = (import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8080").replace(
  /\/+$/,
  "",
);

/* ------------------------------------------------------------------ types */

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
}

export interface RegisterResponse {
  userId: string;
  username: string;
  email: string;
  createdAt: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface TokenResponse {
  accessToken: string;
  tokenType: "Bearer";
  expiresIn: number;
}

export interface QueryRequest {
  conversationId: string | null;
  query: string;
}

export type QueryStatus = "COMPLETED" | "COMPLETED_WITH_WARNINGS";

export interface QueryResponse {
  requestId: string;
  conversationId: string;
  conversationName: string;
  newConversation: boolean;
  answer: string;
  status: QueryStatus;
  executionTimeMs: number;
  retryCount: number;
  warnings: string[];
  persisted: boolean;
}

export type IngestFileStatus = "INGESTED" | "REJECTED" | "FAILED" | "NOT_INGESTED";

export interface IngestFileResult {
  index: number;
  filename: string;
  status: IngestFileStatus;
  documentId?: string;
  chunks?: number;
  vectors?: number;
  storedBytes?: number;
  errorCode?: string;
  message?: string;
}

export interface IngestResponse {
  requestId: string;
  status: string;
  succeeded: number;
  failed: number;
  storageUsedBytes?: number;
  storageLimitBytes?: number;
  executionTimeMs: number;
  files: IngestFileResult[];
}

/** The one error shape every non-2xx response carries. */
export interface ApiErrorBody {
  timestamp?: string;
  status: number;
  error?: string;
  code?: string;
  message?: string;
  path?: string;
  requestId?: string;
  details?: Record<string, unknown>;
}

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: Record<string, unknown>;
  readonly requestId?: string;
  /** Seconds, from the Retry-After header on 429 and BACKEND_BUSY. */
  readonly retryAfter?: number;

  constructor(status: number, code: string, message: string, opts: Partial<Pick<ApiError, "details" | "requestId" | "retryAfter">> = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = opts.details ?? {};
    this.requestId = opts.requestId;
    this.retryAfter = opts.retryAfter;
  }

  /** Field-level validation messages from a 400 VALIDATION_FAILED, keyed by field name. */
  get fieldErrors(): Record<string, string> {
    const out: Record<string, string> = {};
    for (const [k, v] of Object.entries(this.details)) {
      if (typeof v === "string") out[k] = v;
      else if (Array.isArray(v) && typeof v[0] === "string") out[k] = v.join("; ");
    }
    return out;
  }
}

/** Codes produced on the client side, never by the server. */
export const NETWORK_ERROR = "NETWORK_ERROR";
export const ABORTED = "ABORTED";

/* --------------------------------------------------------------- plumbing */

/** 8-64 of [A-Za-z0-9._-], as the API requires; Java and Python both log it. */
export function newRequestId(): string {
  const uuid =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 14)}`;
  return `web-${uuid}`;
}

function parseRetryAfter(value: string | null): number | undefined {
  if (!value) return undefined;
  const secs = Number(value);
  if (Number.isFinite(secs)) return Math.max(0, secs);
  const date = Date.parse(value);
  return Number.isNaN(date) ? undefined : Math.max(0, Math.round((date - Date.now()) / 1000));
}

const FALLBACK_MESSAGES: Record<number, string> = {
  400: "The request was not accepted.",
  401: "Your session has ended. Sign in again.",
  403: "Not allowed.",
  404: "Not found.",
  413: "Too large.",
  429: "Too many attempts. Wait a moment and try again.",
  500: "The server failed unexpectedly.",
  503: "The pipeline is unavailable right now.",
  504: "The pipeline took too long to answer.",
};

export function errorFromBody(status: number, body: unknown, headers?: { get(name: string): string | null }): ApiError {
  const b = (body && typeof body === "object" ? body : {}) as Partial<ApiErrorBody>;
  return new ApiError(
    status,
    b.code ?? b.error?.toUpperCase().replace(/\s+/g, "_") ?? `HTTP_${status}`,
    b.message ?? FALLBACK_MESSAGES[status] ?? `Request failed (${status}).`,
    {
      details: b.details ?? {},
      requestId: b.requestId ?? headers?.get("X-Request-Id") ?? undefined,
      retryAfter: parseRetryAfter(headers?.get("Retry-After") ?? null),
    },
  );
}

async function readJson(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text) return null;
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

/** Called when an authenticated request comes back 401, so the session can be dropped. */
let unauthorizedHandler: (() => void) | null = null;
export function onUnauthorized(handler: (() => void) | null): void {
  unauthorizedHandler = handler;
}

interface RequestOptions {
  token?: string;
  body?: unknown;
  signal?: AbortSignal;
  method?: string;
}

async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const headers: Record<string, string> = { Accept: "application/json", "X-Request-Id": newRequestId() };
  if (opts.body !== undefined) headers["Content-Type"] = "application/json";
  if (opts.token) headers.Authorization = `Bearer ${opts.token}`;

  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}${path}`, {
      method: opts.method ?? (opts.body === undefined ? "GET" : "POST"),
      headers,
      body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
      signal: opts.signal,
      cache: "no-store",
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError(0, ABORTED, "Stopped waiting for the answer.");
    }
    throw new ApiError(0, NETWORK_ERROR, `Cannot reach the Atlas API at ${API_BASE_URL}.`);
  }

  const body = await readJson(res);
  if (!res.ok) {
    const error = errorFromBody(res.status, body, res.headers);
    if (res.status === 401 && opts.token) unauthorizedHandler?.();
    throw error;
  }
  return body as T;
}

/* ------------------------------------------------------------- endpoints */

export const api = {
  register: (req: RegisterRequest) => request<RegisterResponse>("/register", { body: req }),

  login: (req: LoginRequest) => request<TokenResponse>("/login", { body: req }),

  /** Synchronous: on a CPU this takes minutes, so there is deliberately no client timeout. */
  query: (token: string, req: QueryRequest, signal?: AbortSignal) =>
    request<QueryResponse>("/query", { token, body: req, signal }),

  health: async (signal?: AbortSignal): Promise<boolean> => {
    try {
      const body = await request<{ status?: string }>("/actuator/health", { signal });
      return body?.status === "UP";
    } catch {
      return false;
    }
  },

  /**
   * Multipart upload, one `files` part per file. Uses XHR for upload progress. Every status that
   * carries a per-file report (200, 207, 413, 415, 422, 507) resolves; anything else rejects.
   */
  ingest: (
    token: string,
    files: File[],
    opts: { onUploadProgress?: (fraction: number) => void; signal?: AbortSignal } = {},
  ): Promise<{ httpStatus: number; body: IngestResponse }> =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE_URL}/ingest`);
      xhr.setRequestHeader("Authorization", `Bearer ${token}`);
      xhr.setRequestHeader("Accept", "application/json");
      xhr.setRequestHeader("X-Request-Id", newRequestId());

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) opts.onUploadProgress?.(e.loaded / e.total);
      };
      xhr.upload.onload = () => opts.onUploadProgress?.(1);
      xhr.onerror = () => reject(new ApiError(0, NETWORK_ERROR, `Cannot reach the Atlas API at ${API_BASE_URL}.`));
      xhr.onabort = () => reject(new ApiError(0, ABORTED, "Upload cancelled."));
      xhr.onload = () => {
        let body: unknown = null;
        try {
          body = xhr.responseText ? JSON.parse(xhr.responseText) : null;
        } catch {
          body = null;
        }
        const headers = { get: (name: string) => xhr.getResponseHeader(name) };
        if (body && typeof body === "object" && Array.isArray((body as IngestResponse).files)) {
          resolve({ httpStatus: xhr.status, body: body as IngestResponse });
          return;
        }
        if (xhr.status === 401) unauthorizedHandler?.();
        reject(errorFromBody(xhr.status, body, headers));
      };

      opts.signal?.addEventListener("abort", () => xhr.abort(), { once: true });

      const form = new FormData();
      for (const f of files) form.append("files", f, f.name);
      xhr.send(form);
    }),
};

/* --------------------------------------------------- client-side limits */

/** The same limits the server enforces (atlas.upload.*); checked here only to fail fast. */
export const UPLOAD_LIMITS = {
  maxFiles: 10,
  maxFileBytes: 25 * 1024 * 1024,
  maxRequestBytes: 100 * 1024 * 1024,
  extensions: ["pdf", "docx", "pptx", "xlsx", "txt", "md", "html", "htm", "json", "csv"],
} as const;

export const QUERY_MAX_CHARS = 8000;

export const USERNAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.-]{2,31}$/;

export function fileExtension(name: string): string {
  const i = name.lastIndexOf(".");
  return i < 0 ? "" : name.slice(i + 1).toLowerCase();
}

export function formatBytes(n: number | undefined): string {
  if (n === undefined || !Number.isFinite(n)) return "—";
  if (n < 1024) return `${n} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let v = n / 1024;
  let u = 0;
  while (v >= 1024 && u < units.length - 1) {
    v /= 1024;
    u++;
  }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[u]}`;
}

/** A plain-language explanation for each code the API can return, for the UI. */
export function describeError(err: unknown): string {
  if (!(err instanceof ApiError)) return "Something went wrong.";
  switch (err.code) {
    case "BACKEND_BUSY":
      return `The pipeline is busy with other requests.${err.retryAfter ? ` Try again in ${err.retryAfter}s.` : " Try again shortly."}`;
    case "BACKEND_UNAVAILABLE":
      return "The Python pipeline is not reachable, or a model or database failed. Check that `main.py serve` is running.";
    case "BACKEND_TIMEOUT":
      return "The pipeline did not finish before the deadline.";
    case "CONVERSATION_NOT_FOUND":
      return "This conversation no longer exists on the server, or belongs to another account.";
    case "INVALID_CREDENTIALS":
      return "That email and password do not match an account.";
    case "USERNAME_TAKEN":
      return "That username is already taken.";
    case "EMAIL_TAKEN":
      return "An account with that email already exists.";
    default:
      if (err.status === 429) return `${err.message}${err.retryAfter ? ` (retry in ${err.retryAfter}s)` : ""}`;
      return err.message;
  }
}
