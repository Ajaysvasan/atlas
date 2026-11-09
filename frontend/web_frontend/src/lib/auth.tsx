import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api, onUnauthorized, type LoginRequest } from "./api";

/**
 * The session lives in sessionStorage: it survives a reload but not closing the tab, and the API is
 * stateless (no refresh token), so the user signs in again when the token expires.
 */
export interface Session {
  accessToken: string;
  /** The token's `sub`: the user's UUID. Keys this user's local conversation history. */
  userId: string;
  email: string;
  /** Epoch ms. */
  expiresAt: number;
}

const STORAGE_KEY = "atlas.session";
/** Sign out a little before the server would start rejecting the token. */
const EXPIRY_SKEW_MS = 15_000;

/** Reads `sub` from a JWT payload. No verification: the server does that on every call. */
export function decodeSubject(token: string): string | null {
  const part = token.split(".")[1];
  if (!part) return null;
  try {
    const b64 = part.replace(/-/g, "+").replace(/_/g, "/").padEnd(Math.ceil(part.length / 4) * 4, "=");
    const json = JSON.parse(atob(b64)) as { sub?: unknown };
    return typeof json.sub === "string" ? json.sub : null;
  } catch {
    return null;
  }
}

function loadSession(): Session | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const s = JSON.parse(raw) as Session;
    if (!s.accessToken || !s.userId || s.expiresAt - EXPIRY_SKEW_MS <= Date.now()) {
      sessionStorage.removeItem(STORAGE_KEY);
      return null;
    }
    return s;
  } catch {
    return null;
  }
}

function saveSession(s: Session | null): void {
  try {
    if (s) sessionStorage.setItem(STORAGE_KEY, JSON.stringify(s));
    else sessionStorage.removeItem(STORAGE_KEY);
  } catch {
    /* storage blocked: the session still lives in memory for this page */
  }
}

interface AuthContextValue {
  session: Session | null;
  signIn: (req: LoginRequest) => Promise<Session>;
  signOut: (reason?: "expired") => void;
  /** Set when the last sign-out was forced by expiry or a 401, for the sign-in page to explain. */
  endedReason: "expired" | null;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(loadSession);
  const [endedReason, setEndedReason] = useState<"expired" | null>(null);

  const signOut = useCallback((reason?: "expired") => {
    saveSession(null);
    setSession(null);
    setEndedReason(reason ?? null);
  }, []);

  const signIn = useCallback(async (req: LoginRequest) => {
    const token = await api.login(req);
    const userId = decodeSubject(token.accessToken);
    if (!userId) throw new Error("The server returned a token without a subject.");
    const s: Session = {
      accessToken: token.accessToken,
      userId,
      email: req.email.trim().toLowerCase(),
      expiresAt: Date.now() + token.expiresIn * 1000,
    };
    saveSession(s);
    setSession(s);
    setEndedReason(null);
    return s;
  }, []);

  // A 401 on any authenticated call means the token is no longer accepted.
  useEffect(() => {
    onUnauthorized(() => signOut("expired"));
    return () => onUnauthorized(null);
  }, [signOut]);

  // Sign out when the token runs out, rather than on the next failed request.
  useEffect(() => {
    if (!session) return;
    const ms = session.expiresAt - EXPIRY_SKEW_MS - Date.now();
    const id = window.setTimeout(() => signOut("expired"), Math.max(0, ms));
    return () => window.clearTimeout(id);
  }, [session, signOut]);

  const value = useMemo(() => ({ session, signIn, signOut, endedReason }), [session, signIn, signOut, endedReason]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}

/** For pages that only render behind <RequireAuth>. */
export function useSession(): Session {
  const { session } = useAuth();
  if (!session) throw new Error("useSession called without a session");
  return session;
}
