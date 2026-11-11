import { useEffect, useRef, useState } from "react";
import { Link, useLocation } from "react-router";
import { Footer } from "../components/Footer";
import { Header } from "../components/Header";
import { Meridian } from "../components/Meridian";
import { SeaChart } from "../components/SeaChart";
import { SectionLabel } from "../components/SectionLabel";
import { useAuth } from "../lib/auth";
import { revealLines, ScrollTrigger, useMotion, useSmoothScroll } from "../lib/motion";

const STAGES: Array<{ en: string; ja: string; body: string }> = [
  {
    en: "Decomposition",
    ja: "分解",
    body: "The question is split into sub-queries and judged: a quick doubt, or a development that needs a plan.",
  },
  {
    en: "Memory",
    ja: "記憶",
    body: "The conversation so far is recalled, and routed only among your own projects.",
  },
  {
    en: "Retrieval",
    ja: "検索",
    body: "DiskANN search over the corpus, reranked, then checked for sufficiency. When evidence is missing, trusted sources are fetched within the storage quota.",
  },
  {
    en: "Thinking",
    ja: "思考",
    body: "Jobs are planned and run, and the plan is critiqued and revised before anything is written.",
  },
  {
    en: "Generation",
    ja: "生成",
    body: "The answer is written from the evidence by a local Qwen2.5-3B model. Nothing leaves the machine.",
  },
  {
    en: "Persistence",
    ja: "保存",
    body: "The turn is stored in memory and the model state checkpointed, so the next question starts warm.",
  },
];

const CREDITS: Array<[string, string]> = [
  ["Passwords", "Argon2id, salted per user"],
  ["Sessions", "Signed tokens, one hour, no cookies"],
  ["Uploads", "Private to the account that sent them"],
  ["Memory", "Routed only among your own projects"],
  ["Conversations", "Answer to their owner, and nobody else"],
  ["Transport", "gRPC on loopback, with a service token"],
];

const SECTIONS = ["introduction", "pipeline", "privacy", "access"];
const TICKS = ["00 DEPART", "01 VOYAGE", "02 PORTS", "03 WATERS", "04 SAIL"];

/**
 * Your own key art, when you have it: put a 2560×1440 image (subject right of centre, transparent
 * or dark on the left) in public/ and set its path here, e.g. "/key-visual.webp". The sea chart
 * then sits over it, dimmed, so the art leads.
 */
const HERO_ART: string | null = null;

export function Landing() {
  const root = useRef<HTMLDivElement>(null);
  const { session } = useAuth();
  const location = useLocation();
  const [active, setActive] = useState(0);

  useSmoothScroll();

  useMotion(root, (el) => {
    revealLines(el);
    SECTIONS.forEach((id, i) => {
      ScrollTrigger.create({
        trigger: `#${id}`,
        start: "top 55%",
        end: "bottom 55%",
        onToggle: (self) => self.isActive && setActive(i + 1),
        onLeaveBack: () => i === 0 && setActive(0),
      });
    });
  });

  useEffect(() => {
    if (!location.hash) return;
    document.getElementById(location.hash.slice(1))?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [location.hash]);

  const primary = session ? { to: "/workspace", label: "Open workspace" } : { to: "/signup", label: "Create account" };

  return (
    <div ref={root} className="relative">
      <Header variant="public" />

      {/* The meridian: fixed along the left edge, one tick per section. */}
      <Meridian
        ticks={TICKS}
        active={active}
        className="fixed bottom-[12vh] left-[calc(var(--gutter)*0.5)] top-[18vh] z-30 hidden lg:block"
      />

      <main>
        {/* HERO / KEY VISUAL */}
        <section className="relative isolate flex min-h-[100svh] flex-col overflow-hidden" aria-labelledby="hero-title">
          <div className="absolute inset-0 -z-10 bg-[radial-gradient(110%_90%_at_72%_35%,var(--base-2)_0%,var(--base)_72%)]" />
          {HERO_ART && (
            <img src={HERO_ART} alt="" className="absolute inset-0 -z-10 h-full w-full object-cover object-right" fetchPriority="high" />
          )}
          <div className="pointer-events-none absolute inset-0 -z-10">
            <SeaChart className={`h-full w-full ${HERO_ART ? "opacity-40" : "opacity-95 max-md:opacity-45"}`} />
          </div>
          {/* scrims: keep the title and the call to action legible over the chart */}
          <div className="absolute inset-y-0 left-0 -z-10 w-[62%] bg-gradient-to-r from-base via-base/70 to-transparent [mask-image:linear-gradient(to_bottom,transparent_18%,black_45%)] max-md:w-full max-md:[mask-image:none]" />
          <div className="absolute inset-x-0 bottom-0 -z-10 h-[58%] bg-gradient-to-t from-base via-base/85 to-transparent" />

          <div className="mt-auto grid grid-cols-12 gap-x-gutter px-gutter pb-[10vh] pt-28">
            <div className="col-span-12 lg:col-span-9 lg:col-start-2">
              <p className="label mb-6" data-fade="now" data-delay="0.2">
                <span className="text-brass">00°00′N · 00°00′E</span>
                <span className="mx-3 text-ink-faint">—</span>
                Local retrieval-augmented generation
              </p>
              <h1
                id="hero-title"
                data-reveal="now"
                className="display w-fit whitespace-nowrap text-[clamp(6.5rem,22vw,17.5rem)] leading-[0.84]"
              >
                ATLAS
              </h1>
              <div className="mt-8 grid gap-6 md:grid-cols-[minmax(0,34rem)_auto] md:items-end md:gap-12">
                <div>
                  <p data-reveal="now" data-delay="0.25" className="text-[clamp(1.25rem,2.4vw,1.9rem)] font-bold leading-[1.3]">
                    Ask your own documents. Answered on your own machine.
                  </p>
                  <p lang="ja" data-fade="now" data-delay="0.6" className="mt-3 text-sm tracking-[0.12em] text-ink-dim">
                    あなたの資料に、あなたの手元で答える。
                  </p>
                </div>
                <div className="flex flex-wrap items-center gap-4" data-fade="now" data-delay="0.8">
                  <Link to={primary.to} className="btn btn-solid">
                    {primary.label}
                    <span aria-hidden="true">→</span>
                  </Link>
                  {!session && (
                    <Link to="/signin" className="label link">
                      Sign in
                    </Link>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between gap-4 border-t border-rule px-gutter py-4">
            <p className="label truncate">Qwen2.5-3B · DiskANN · PostgreSQL</p>
            <p className="label hidden md:block">v0.1 — Final year project</p>
            <p className="label hidden sm:block">Scroll</p>
          </div>
        </section>

        {/* 01 — INTRODUCTION */}
        <section id="introduction" className="grid scroll-mt-20 grid-cols-12 gap-x-gutter px-gutter py-[18vh]">
          <SectionLabel index="01" en="The voyage" ja="航海" className="col-span-12 mb-14 lg:col-span-10 lg:col-start-2" />
          <blockquote
            data-reveal
            className="display col-span-12 text-[clamp(2.2rem,5.2vw,4.8rem)] leading-[1.02] lg:col-span-9 lg:col-start-2"
          >
            Every answer is traced through memory, retrieval and a critique before a word of it is written.
          </blockquote>
          <div className="col-span-12 mt-16 grid gap-10 text-ink-dim md:grid-cols-2 lg:col-span-7 lg:col-start-5">
            <p data-fade>
              Atlas reads the documents you give it and the reference pages it was built with, and keeps a memory of
              every conversation. A question goes through the same pipeline each time, from decomposition to a written
              answer, on one model running locally.
            </p>
            <p data-fade>
              Each answer is planned, checked against the evidence and revised before it is written: seconds on a GPU,
              minutes on a CPU. What you upload stays private to your account, and only your own queries can retrieve
              it.
            </p>
          </div>
        </section>

        {/* 02 — PIPELINE */}
        <section id="pipeline" className="scroll-mt-20 border-t border-rule px-gutter py-[18vh]">
          <div className="grid grid-cols-12 gap-x-gutter">
            <SectionLabel index="02" en="Ports of call" ja="寄港地" className="col-span-12 mb-14 lg:col-span-10 lg:col-start-2" />
            <h2
              data-reveal
              className="display col-span-12 mb-16 text-[clamp(2.8rem,7vw,6.5rem)] leading-[0.92] lg:col-span-8 lg:col-start-2"
            >
              Six stages, one model.
            </h2>
            <ol className="col-span-12 border-t border-rule lg:col-span-10 lg:col-start-2">
              {STAGES.map((s, i) => (
                <li
                  key={s.en}
                  data-fade
                  className="group grid grid-cols-[3rem_1fr] gap-x-6 border-b border-rule py-7 transition-transform duration-700 ease-[var(--ease-out)] hover:translate-x-2 md:grid-cols-[4rem_minmax(0,21rem)_1fr_2rem] md:items-baseline"
                >
                  <span className="flex items-center gap-3 font-mono text-xs tracking-[0.2em] text-accent-ink">
                    <span className="port" aria-hidden="true" />
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <h3 className="display text-3xl md:text-4xl">
                    {s.en}
                    <span lang="ja" className="ml-3 align-middle font-body text-sm font-normal normal-case tracking-normal text-ink-faint">
                      {s.ja}
                    </span>
                  </h3>
                  <p className="col-start-2 mt-2 max-w-[56ch] text-ink-dim md:col-start-3 md:mt-0">{s.body}</p>
                  <span
                    aria-hidden="true"
                    className="hidden -translate-x-2 text-accent-ink opacity-0 transition-all duration-700 group-hover:translate-x-0 group-hover:opacity-100 md:block"
                  >
                    →
                  </span>
                </li>
              ))}
            </ol>
          </div>
        </section>

        {/* 03 — PRIVACY: set like end credits */}
        <section id="privacy" className="scroll-mt-20 border-t border-rule px-gutter py-[18vh]">
          <div className="grid grid-cols-12 gap-x-gutter">
            <SectionLabel index="03" en="Own waters" ja="領海" className="col-span-12 mb-14 lg:col-span-10 lg:col-start-2" />
            <h2
              data-reveal
              className="display col-span-12 text-[clamp(2.8rem,7vw,6.5rem)] leading-[0.92] lg:col-span-5 lg:col-start-2"
            >
              Yours stays yours.
            </h2>
            <dl className="col-span-12 mt-14 space-y-6 lg:col-span-5 lg:col-start-7 lg:mt-3">
              {CREDITS.map(([role, name]) => (
                <div key={role} data-fade className="grid grid-cols-[minmax(0,11rem)_1fr] gap-6">
                  <dt className="label pt-1.5 text-right">{role}</dt>
                  <dd className="text-lg leading-snug">{name}</dd>
                </div>
              ))}
            </dl>
          </div>
        </section>

        {/* 04 — ACCESS */}
        <section id="access" className="scroll-mt-20 border-t border-rule px-gutter py-[20vh]">
          <div className="grid grid-cols-12 gap-x-gutter">
            <SectionLabel index="04" en="Set sail" ja="出航" className="col-span-12 mb-14 lg:col-span-10 lg:col-start-2" />
            <p
              data-reveal
              className="display col-span-12 text-[clamp(3.6rem,12vw,11.5rem)] leading-[0.86] lg:col-span-10 lg:col-start-2"
            >
              Ask the first question.
            </p>
            <div data-fade className="col-span-12 mt-12 flex flex-wrap items-center gap-6 lg:col-span-10 lg:col-start-2">
              <Link to={primary.to} className="btn btn-solid">
                {primary.label}
                <span aria-hidden="true">→</span>
              </Link>
              {!session && (
                <Link to="/signin" className="btn">
                  Sign in
                </Link>
              )}
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}
