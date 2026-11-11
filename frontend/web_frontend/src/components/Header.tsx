import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Link, NavLink, useLocation } from "react-router";
import { useAuth } from "../lib/auth";
import { EASE_WIPE, gsap, MOTION_OK } from "../lib/motion";
import { ApiStatus } from "./ApiStatus";

interface NavItem {
  to: string;
  en: string;
  ja: string;
}

const PUBLIC_NAV: NavItem[] = [
  { to: "/#introduction", en: "Introduction", ja: "概要" },
  { to: "/#pipeline", en: "Pipeline", ja: "仕組み" },
  { to: "/#privacy", en: "Privacy", ja: "分離" },
];

const APP_NAV: NavItem[] = [
  { to: "/workspace", en: "Workspace", ja: "会話" },
  { to: "/documents", en: "Documents", ja: "資料" },
];

export function Wordmark() {
  return (
    <Link to="/" className="group flex items-baseline gap-3" aria-label="Atlas, home">
      <span className="display text-[23px] leading-none tracking-[0.04em] text-ink">ATLAS</span>
      <span lang="ja" className="hidden font-body text-[11px] tracking-[0.25em] text-ink-faint sm:inline">
        アトラス
      </span>
    </Link>
  );
}

/** Fixed chrome: wordmark left, small uppercase labels right, and a full-screen menu on mobile. */
export function Header({ variant }: { variant: "public" | "app" }) {
  const { session, signOut } = useAuth();
  const [open, setOpen] = useState(false);
  const location = useLocation();
  const nav = variant === "app" ? APP_NAV : PUBLIC_NAV;

  useEffect(() => setOpen(false), [location.pathname, location.hash]);

  return (
    <header
      className={`${variant === "public" ? "fixed" : "sticky"} inset-x-0 top-0 z-40 border-b border-rule bg-base/80 backdrop-blur-md`}
    >
      <div className="flex h-14 items-center justify-between px-gutter">
        <Wordmark />

        <nav aria-label="Primary" className="hidden items-center gap-8 md:flex">
          {nav.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `label link pb-0.5 ${isActive && variant === "app" ? "text-ink [background-size:100%_1px]" : ""}`
              }
            >
              {item.en}
            </NavLink>
          ))}
          <span className="h-4 w-px bg-rule-strong" aria-hidden="true" />
          <ApiStatus />
          {session ? (
            <>
              {variant === "public" && (
                <Link to="/workspace" className="label link text-ink">
                  Open workspace
                </Link>
              )}
              {variant === "app" && (
                <span className="label max-w-[18ch] truncate normal-case tracking-[0.08em]" title={session.email}>
                  {session.email}
                </span>
              )}
              <button type="button" onClick={() => signOut()} className="label link">
                Sign out
              </button>
            </>
          ) : (
            <>
              <Link to="/signin" className="label link">
                Sign in
              </Link>
              <Link to="/signup" className="label link text-ink">
                Create account
              </Link>
            </>
          )}
        </nav>

        <button
          type="button"
          className="label flex items-center gap-3 md:hidden"
          aria-expanded={open}
          aria-controls="overlay-menu"
          onClick={() => setOpen(true)}
        >
          Menu
          <span className="flex flex-col gap-[5px]" aria-hidden="true">
            <span className="block h-px w-6 bg-ink" />
            <span className="block h-px w-4 self-end bg-accent" />
          </span>
        </button>
      </div>

      {/* Portalled: the header's backdrop-filter would otherwise become the overlay's containing block. */}
      {open &&
        createPortal(
          <OverlayMenu
            nav={variant === "public" && session ? [...nav, { to: "/workspace", en: "Workspace", ja: "会話" }] : nav}
            onClose={() => setOpen(false)}
          />,
          document.body,
        )}
    </header>
  );
}

function OverlayMenu({ nav, onClose }: { nav: NavItem[]; onClose: () => void }) {
  const { session, signOut } = useAuth();
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    const previous = document.activeElement as HTMLElement | null;
    root.querySelector<HTMLElement>("a, button")?.focus();

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key !== "Tab") return;
      const focusable = Array.from(root.querySelectorAll<HTMLElement>("a, button"));
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";

    const mm = gsap.matchMedia();
    mm.add(MOTION_OK, () => {
      gsap
        .timeline()
        .fromTo(root.querySelector("[data-wipe]"), { scaleY: 0 }, { scaleY: 1, transformOrigin: "top", duration: 0.55, ease: EASE_WIPE })
        .from(root.querySelectorAll("[data-item]"), { yPercent: 110, duration: 0.8, ease: "power4.out", stagger: 0.06 }, "-=0.15");
    });

    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
      mm.revert();
      previous?.focus();
    };
  }, [onClose]);

  const linkCls = "display block whitespace-nowrap text-[clamp(2.4rem,10.5vw,5rem)] leading-[1.05]";

  return (
    <div
      ref={ref}
      id="overlay-menu"
      role="dialog"
      aria-modal="true"
      aria-label="Menu"
      className="fixed inset-0 z-50 flex flex-col bg-base px-gutter"
    >
      <div data-wipe className="route-line absolute bottom-0 left-6 top-14 w-[2px]" aria-hidden="true" />
      <div className="flex h-14 items-center justify-between">
        <Wordmark />
        <button type="button" className="label" onClick={onClose}>
          Close
        </button>
      </div>
      <nav aria-label="Menu" className="mt-[10vh] flex flex-col gap-5 pl-8">
        {nav.map((item, i) => (
          <span key={item.to} className="mask">
            <Link data-item to={item.to} className={linkCls} onClick={onClose}>
              <span className="mr-4 align-top font-mono text-xs normal-case tracking-[0.2em] text-accent-ink">0{i + 1}</span>
              {item.en}
              <span lang="ja" className="ml-4 align-middle font-body text-sm font-normal normal-case tracking-normal text-ink-faint">
                {item.ja}
              </span>
            </Link>
          </span>
        ))}
        <span className="mask mt-6">
          {session ? (
            <button data-item type="button" className="label link" onClick={() => signOut()}>
              Sign out — {session.email}
            </button>
          ) : (
            <span data-item className="flex gap-8">
              <Link to="/signin" className="label link">
                Sign in
              </Link>
              <Link to="/signup" className="label link text-ink">
                Create account
              </Link>
            </span>
          )}
        </span>
      </nav>
      <div className="mt-auto pb-8 pl-8">
        <ApiStatus />
      </div>
    </div>
  );
}
