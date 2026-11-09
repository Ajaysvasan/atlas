import { useRef, useState, type SubmitEvent, type ReactNode } from "react";
import { Link, Navigate, useNavigate, useSearchParams } from "react-router";
import { Wordmark } from "../components/Header";
import { Meridian } from "../components/Meridian";
import { api, ApiError, describeError, USERNAME_PATTERN } from "../lib/api";
import { useAuth } from "../lib/auth";
import { revealLines, useMotion } from "../lib/motion";

/** Only same-site paths are followed after sign-in, never an absolute URL. */
function safeNext(raw: string | null): string {
  if (!raw || !raw.startsWith("/") || raw.startsWith("//")) return "/workspace";
  return raw;
}

function AuthLayout({
  index,
  en,
  ja,
  title,
  children,
  aside,
}: {
  index: string;
  en: string;
  ja: string;
  title: string;
  children: ReactNode;
  aside: ReactNode;
}) {
  const root = useRef<HTMLDivElement>(null);
  useMotion(root, revealLines);

  return (
    <div ref={root} className="grid min-h-[100svh] grid-cols-1 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
      {/* Left: the title panel, with the meridian running down its edge */}
      <div className="relative flex flex-col justify-between overflow-hidden border-rule px-gutter py-8 lg:border-r">
        <Wordmark />
        <div className="my-16 lg:my-0">
          <p className="label mb-6">
            <span className="text-accent">{index}</span>
            <span className="mx-3 text-ink-faint">—</span>
            {en} <span className="text-ink-faint">/</span>{" "}
            <span lang="ja" className="tracking-[0.1em]">
              {ja}
            </span>
          </p>
          <h1
            data-reveal="now"
            className="font-display text-[clamp(2.75rem,5.4vw,5.75rem)] font-extrabold leading-[0.92] tracking-[-0.045em]"
          >
            {title}
          </h1>
        </div>
        <div className="hidden text-ink-dim lg:block">{aside}</div>
        <Meridian className="absolute bottom-0 right-0 top-0 hidden lg:block" />
      </div>

      {/* Right: the form */}
      <div className="flex items-center px-gutter pb-16 lg:py-16">
        <div className="w-full max-w-md lg:ml-[12%]" data-fade="now" data-delay="0.3">
          {children}
        </div>
      </div>
    </div>
  );
}

function Field({
  id,
  label,
  error,
  hint,
  ...input
}: { id: string; label: string; error?: string; hint?: string } & React.InputHTMLAttributes<HTMLInputElement>) {
  const describedBy = error ? `${id}-error` : hint ? `${id}-hint` : undefined;
  return (
    <div>
      <label htmlFor={id} className="label block">
        {label}
      </label>
      <input id={id} name={id} className="field" aria-invalid={error ? true : undefined} aria-describedby={describedBy} {...input} />
      {error ? (
        <p id={`${id}-error`} className="mt-2 text-sm text-danger">
          {error}
        </p>
      ) : hint ? (
        <p id={`${id}-hint`} className="mt-2 text-xs text-ink-faint">
          {hint}
        </p>
      ) : null}
    </div>
  );
}

function FormError({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <p role="alert" className="border-l border-danger py-1 pl-4 text-sm text-ink">
      {message}
    </p>
  );
}

/* ---------------------------------------------------------------- sign in */

export function SignIn() {
  const { session, signIn, endedReason } = useAuth();
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const [email, setEmail] = useState(params.get("email") ?? "");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const next = safeNext(params.get("next"));
  const justRegistered = params.get("registered") === "1";

  if (session) return <Navigate to={next} replace />;

  async function onSubmit(e: SubmitEvent) {
    e.preventDefault();
    if (!email.trim() || !password) {
      setError("Enter your email and password.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await signIn({ email: email.trim(), password });
      navigate(next, { replace: true });
    } catch (err) {
      setError(describeError(err));
      setBusy(false);
    }
  }

  return (
    <AuthLayout
      index="01"
      en="Sign in"
      ja="ログイン"
      title="Welcome back."
      aside={<p className="max-w-sm text-sm">Sessions last one hour. Your conversation list is kept in this browser.</p>}
    >
      <form onSubmit={onSubmit} noValidate className="space-y-9">
        {justRegistered && (
          <p role="status" className="border-l border-accent py-1 pl-4 text-sm">
            Account created. Sign in to continue.
          </p>
        )}
        {endedReason === "expired" && !justRegistered && (
          <p role="status" className="border-l border-accent py-1 pl-4 text-sm">
            Your session ended. Sign in again.
          </p>
        )}
        <Field
          id="email"
          label="Email"
          type="email"
          autoComplete="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          maxLength={254}
          required
        />
        <Field
          id="password"
          label="Password"
          type="password"
          autoComplete="current-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          maxLength={128}
          required
        />
        <FormError message={error} />
        <div className="flex flex-wrap items-center justify-between gap-6 pt-2">
          <button type="submit" className="btn btn-solid" disabled={busy}>
            {busy ? "Signing in" : "Sign in"}
            <span aria-hidden="true">→</span>
          </button>
          <Link to={`/signup${params.get("next") ? `?next=${encodeURIComponent(next)}` : ""}`} className="label link">
            Create an account
          </Link>
        </div>
      </form>
    </AuthLayout>
  );
}

/* ---------------------------------------------------------------- sign up */

type Fields = "username" | "email" | "password" | "confirm";

export function validateRegistration(v: Record<Fields, string>): Partial<Record<Fields, string>> {
  const errors: Partial<Record<Fields, string>> = {};
  const username = v.username.trim();
  if (!USERNAME_PATTERN.test(username))
    errors.username = "3–32 characters: letters, digits, '.', '_' or '-', starting with a letter or digit.";
  const email = v.email.trim();
  if (!email || email.length > 254 || !/^[^\s@]+@[^\s@]+$/.test(email)) errors.email = "Enter a valid email address.";
  if (v.password.length < 8 || v.password.length > 128) errors.password = "8–128 characters.";
  if (v.confirm !== v.password) errors.confirm = "The passwords do not match.";
  return errors;
}

export function SignUp() {
  const { session } = useAuth();
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const [values, setValues] = useState<Record<Fields, string>>({ username: "", email: "", password: "", confirm: "" });
  const [errors, setErrors] = useState<Partial<Record<Fields, string>>>({});
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  if (session) return <Navigate to="/workspace" replace />;

  const set = (k: Fields) => (e: React.ChangeEvent<HTMLInputElement>) => {
    setValues((v) => ({ ...v, [k]: e.target.value }));
    if (errors[k]) setErrors((x) => ({ ...x, [k]: undefined }));
  };

  async function onSubmit(e: SubmitEvent) {
    e.preventDefault();
    const found = validateRegistration(values);
    setErrors(found);
    setFormError(null);
    if (Object.keys(found).length) return;
    setBusy(true);
    try {
      const res = await api.register({ username: values.username.trim(), email: values.email.trim(), password: values.password });
      const q = new URLSearchParams({ registered: "1", email: res.email });
      const next = params.get("next");
      if (next) q.set("next", next);
      navigate(`/signin?${q}`, { replace: true });
    } catch (err) {
      if (err instanceof ApiError && err.code === "VALIDATION_FAILED") {
        setErrors(err.fieldErrors);
      } else if (err instanceof ApiError && err.code === "USERNAME_TAKEN") {
        setErrors({ username: describeError(err) });
      } else if (err instanceof ApiError && err.code === "EMAIL_TAKEN") {
        setErrors({ email: describeError(err) });
      } else {
        setFormError(describeError(err));
      }
      setBusy(false);
    }
  }

  return (
    <AuthLayout
      index="00"
      en="Create account"
      ja="新規登録"
      title="Open an account."
      aside={
        <p className="max-w-sm text-sm">
          Documents you upload are private to this account. Passwords are hashed with Argon2id and never stored as typed.
        </p>
      }
    >
      <form onSubmit={onSubmit} noValidate className="space-y-9">
        <Field
          id="username"
          label="Username"
          autoComplete="username"
          value={values.username}
          onChange={set("username")}
          error={errors.username}
          hint="3–32 characters: letters, digits, '.', '_' or '-'."
          maxLength={32}
          required
        />
        <Field
          id="email"
          label="Email"
          type="email"
          autoComplete="email"
          value={values.email}
          onChange={set("email")}
          error={errors.email}
          maxLength={254}
          required
        />
        <Field
          id="password"
          label="Password"
          type="password"
          autoComplete="new-password"
          value={values.password}
          onChange={set("password")}
          error={errors.password}
          hint="8–128 characters."
          maxLength={128}
          required
        />
        <Field
          id="confirm"
          label="Confirm password"
          type="password"
          autoComplete="new-password"
          value={values.confirm}
          onChange={set("confirm")}
          error={errors.confirm}
          maxLength={128}
          required
        />
        <FormError message={formError} />
        <div className="flex flex-wrap items-center justify-between gap-6 pt-2">
          <button type="submit" className="btn btn-solid" disabled={busy}>
            {busy ? "Creating" : "Create account"}
            <span aria-hidden="true">→</span>
          </button>
          <Link to="/signin" className="label link">
            I have an account
          </Link>
        </div>
      </form>
    </AuthLayout>
  );
}
