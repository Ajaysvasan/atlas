import { Link } from "react-router";
import { Header } from "../components/Header";
import { Meridian } from "../components/Meridian";

export function NotFound() {
  return (
    <div className="relative min-h-[100svh]">
      <Header variant="public" />
      <Meridian className="absolute bottom-0 left-[calc(var(--gutter)*0.5)] top-14 hidden lg:block" />
      <main className="flex min-h-[100svh] flex-col justify-end px-gutter pb-[14vh] lg:pl-[calc(var(--gutter)*2.5)]">
        <p className="label mb-6">
          <span className="text-accent">404</span>
          <span className="mx-3 text-ink-faint">—</span>
          Off the map <span className="text-ink-faint">/</span> <span lang="ja">見つかりません</span>
        </p>
        <h1 className="font-display text-[clamp(3.5rem,13vw,12rem)] font-extrabold leading-[0.86] tracking-[-0.045em]">
          No such page.
        </h1>
        <Link to="/" className="btn mt-12 self-start">
          Back to Atlas
        </Link>
      </main>
    </div>
  );
}
