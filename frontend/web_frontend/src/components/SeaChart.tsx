import { useRef } from "react";
import { gsap, useMotion } from "../lib/motion";

/**
 * The hero's key visual: an original sea chart, generated rather than traced. Invented islands (seeded,
 * so the same every load), rhumb lines from a compass rose, depth soundings, and the voyage route
 * through six ports — one per pipeline stage. The route draws itself in on load; scrolling drifts the
 * chart a little. With reduced motion it is drawn complete and stays still.
 *
 * Real key art can replace or sit behind it: see HERO_ART in pages/Landing.tsx.
 */
const W = 1600;
const H = 1000;

type Pt = [number, number];

const ISLANDS: Array<{ c: Pt; r: number; seed: number }> = [
  { c: [210, 330], r: 70, seed: 67 },
  { c: [540, 250], r: 50, seed: 37 },
  { c: [930, 190], r: 95, seed: 11 },
  { c: [1500, 210], r: 60, seed: 41 },
  { c: [1560, 760], r: 120, seed: 23 },
];

const ROSE: Pt = [1265, 600];
const ROSE_R = 130;

// Ports stay clear of the hero's text, which sits over the lower-left of the chart. A port near the
// right edge puts its label on the left.
const PORTS: Array<{ at: Pt; label: string; left?: boolean }> = [
  { at: [360, 140], label: "Decomposition" },
  { at: [700, 120], label: "Memory" },
  { at: [1000, 330], label: "Retrieval" },
  { at: [1340, 330], label: "Thinking" },
  { at: [1525, 415], label: "Generation", left: true },
  { at: [1440, 915], label: "Persistence", left: true },
];
const ROUTE: Pt[] = [[-60, 60], ...PORTS.map((p) => p.at), [1380, 1080]];

/** mulberry32: a small seeded PRNG, so the chart is identical on every render. */
function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A closed smooth path through points (Catmull-Rom, as cubic Béziers). */
function closedPath(pts: Pt[]): string {
  const n = pts.length;
  let d = `M${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`;
  for (let i = 0; i < n; i++) {
    const [p0, p1, p2, p3] = [pts[(i - 1 + n) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]];
    const c1: Pt = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
    const c2: Pt = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
    d += ` C${c1[0].toFixed(1)} ${c1[1].toFixed(1)} ${c2[0].toFixed(1)} ${c2[1].toFixed(1)} ${p2[0].toFixed(1)} ${p2[1].toFixed(1)}`;
  }
  return `${d} Z`;
}

/** An open smooth path through points. */
function openPath(pts: Pt[]): string {
  let d = `M${pts[0][0]} ${pts[0][1]}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)];
    const [p1, p2] = [pts[i], pts[i + 1]];
    const p3 = pts[Math.min(pts.length - 1, i + 2)];
    const c1: Pt = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
    const c2: Pt = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
    d += ` C${c1[0].toFixed(1)} ${c1[1].toFixed(1)} ${c2[0].toFixed(1)} ${c2[1].toFixed(1)} ${p2[0]} ${p2[1]}`;
  }
  return d;
}

/** A coastline: a circle pushed in and out by a few seeded sine waves, so it reads as land, not a blob. */
function coastline(c: Pt, r: number, seed: number, scale = 1): string {
  const rand = rng(seed);
  const waves = [2, 3, 5, 7].map((k) => ({ k, amp: 0.06 + rand() * 0.12, phase: rand() * Math.PI * 2 }));
  const pts: Pt[] = [];
  const n = 36;
  for (let i = 0; i < n; i++) {
    const t = (i / n) * Math.PI * 2;
    const f = 1 + waves.reduce((s, w) => s + w.amp * Math.sin(w.k * t + w.phase), 0);
    const rr = r * f * scale;
    pts.push([c[0] + Math.cos(t) * rr, c[1] + Math.sin(t) * rr * 0.82]);
  }
  return closedPath(pts);
}

function soundings() {
  const rand = rng(97);
  const out: Array<{ x: number; y: number; d: number }> = [];
  for (let y = 60; y < H; y += 74) {
    for (let x = 40; x < W; x += 92) {
      const px = x + (rand() - 0.5) * 50;
      const py = y + (rand() - 0.5) * 40;
      if (rand() < 0.55) continue;
      if (ISLANDS.some((i) => Math.hypot(px - i.c[0], py - i.c[1]) < i.r * 1.45)) continue;
      if (Math.hypot(px - ROSE[0], py - ROSE[1]) < ROSE_R + 50) continue;
      if (PORTS.some((p) => Math.abs(py - (p.at[1] - 18)) < 24 && (p.left ? px > p.at[0] - 270 && px < p.at[0] + 20 : px > p.at[0] - 20 && px < p.at[0] + 270))) continue;
      out.push({ x: px, y: py, d: 3 + Math.floor(rand() * 46) });
    }
  }
  return out;
}

/** One half of a compass point: centre, tip, and one side of the base. */
function half(theta: number, length: number, width: number, side: 1 | -1): string {
  const tip: Pt = [Math.sin(theta) * length, -Math.cos(theta) * length];
  const base: Pt = [Math.cos(theta) * width * side, Math.sin(theta) * width * side];
  return `M0 0 L${tip[0].toFixed(1)} ${tip[1].toFixed(1)} L${base[0].toFixed(1)} ${base[1].toFixed(1)} Z`;
}

const SOUNDINGS = soundings();
const DEG = Math.PI / 180;

function CompassRose() {
  return (
    <g transform={`translate(${ROSE[0]} ${ROSE[1]})`}>
      <circle r={ROSE_R} stroke="var(--brass)" strokeWidth="1.2" opacity="0.8" />
      <circle r={ROSE_R - 8} stroke="var(--brass)" strokeWidth="0.6" opacity="0.6" />
      <circle r={52} stroke="var(--rule-strong)" strokeWidth="0.8" />
      {Array.from({ length: 72 }, (_, i) => {
        const a = i * 5 * DEG;
        const outer = i % 6 === 0 ? ROSE_R + 7 : ROSE_R;
        return (
          <line
            key={i}
            x1={Math.sin(a) * (ROSE_R - 8)}
            y1={-Math.cos(a) * (ROSE_R - 8)}
            x2={Math.sin(a) * outer}
            y2={-Math.cos(a) * outer}
            stroke="var(--brass)"
            strokeWidth={i % 6 === 0 ? 1.2 : 0.6}
            opacity="0.7"
          />
        );
      })}
      {/* eight minor points */}
      {Array.from({ length: 8 }, (_, i) => {
        const a = (22.5 + i * 45) * DEG;
        return <line key={i} x2={Math.sin(a) * 70} y2={-Math.cos(a) * 70} stroke="var(--rule-strong)" strokeWidth="1" />;
      })}
      {/* intercardinal points */}
      {[45, 135, 225, 315].map((d) => (
        <g key={d}>
          <path d={half(d * DEG, 88, 13, 1)} fill="var(--ink)" opacity="0.35" />
          <path d={half(d * DEG, 88, 13, -1)} fill="var(--base)" stroke="var(--ink-dim)" strokeWidth="0.8" />
        </g>
      ))}
      {/* cardinal points; north in vermilion */}
      {[0, 90, 180, 270].map((d) => {
        const north = d === 0;
        return (
          <g key={d}>
            <path d={half(d * DEG, 124, 19, 1)} fill={north ? "var(--accent)" : "var(--ink)"} opacity={north ? 1 : 0.8} />
            <path
              d={half(d * DEG, 124, 19, -1)}
              fill="var(--base)"
              stroke={north ? "var(--accent)" : "var(--ink-dim)"}
              strokeWidth="1"
            />
          </g>
        );
      })}
      <circle r={6} fill="var(--base)" stroke="var(--ink)" strokeWidth="1.2" />
      {(["N", "E", "S", "W"] as const).map((l, i) => {
        const a = i * 90 * DEG;
        return (
          <text
            key={l}
            x={Math.sin(a) * (ROSE_R + 26)}
            y={-Math.cos(a) * (ROSE_R + 26) + 7}
            textAnchor="middle"
            fill={l === "N" ? "var(--accent-ink)" : "var(--ink-dim)"}
            fontFamily="var(--font-display)"
            fontSize="20"
          >
            {l}
          </text>
        );
      })}
    </g>
  );
}

export function SeaChart({ className = "" }: { className?: string }) {
  const ref = useRef<SVGSVGElement>(null);

  useMotion(ref, (svg) => {
    const route = svg.querySelector("[data-route-mask]");
    const ports = svg.querySelectorAll("[data-port]");
    gsap.from(svg.querySelectorAll("[data-coast]"), { opacity: 0, duration: 1.6, ease: "power2.out", stagger: 0.08 });
    if (route) gsap.from(route, { strokeDashoffset: 1, duration: 3, ease: "power3.inOut", delay: 0.3 });
    gsap.from(ports, { opacity: 0, duration: 0.6, ease: "power2.out", stagger: 0.38, delay: 0.75 });
    gsap.to(svg, {
      yPercent: 5,
      ease: "none",
      scrollTrigger: { trigger: svg, start: "top top", end: "bottom top", scrub: 1 },
    });
  });

  const routePath = openPath(ROUTE);

  return (
    <svg
      ref={ref}
      viewBox={`0 0 ${W} ${H}`}
      // Anchored right: on narrow screens the crop falls on the left, under the title and its scrim.
      preserveAspectRatio="xMaxYMid slice"
      className={className}
      role="img"
      aria-label="Sea chart: a dashed route through six ports, one for each stage of the pipeline"
      fill="none"
    >
      <defs>
        <mask id="route-reveal" maskUnits="userSpaceOnUse" x="0" y="0" width={W} height={H}>
          <path
            data-route-mask
            d={routePath}
            stroke="#fff"
            strokeWidth="14"
            pathLength={1}
            strokeDasharray="1"
            strokeDashoffset="0"
          />
        </mask>
      </defs>

      {/* graticule, with degree labels along the top and right edges */}
      <g stroke="var(--rule)" strokeWidth="0.7" opacity="0.7">
        {Array.from({ length: W / 100 + 1 }, (_, i) => (
          <line key={`v${i}`} x1={i * 100} x2={i * 100} y1={0} y2={H} />
        ))}
        {Array.from({ length: H / 100 + 1 }, (_, i) => (
          <line key={`h${i}`} y1={i * 100} y2={i * 100} x1={0} x2={W} />
        ))}
      </g>
      <g fill="var(--brass)" fontFamily="var(--font-mono)" fontSize="11" letterSpacing="2" opacity="0.75">
        {Array.from({ length: 7 }, (_, i) => (
          <text key={`lon${i}`} x={200 + i * 200 + 6} y={90}>
            {120 + i * 5}°E
          </text>
        ))}
        {Array.from({ length: 4 }, (_, i) => (
          <text key={`lat${i}`} x={W - 70} y={200 + i * 200 - 6}>
            {30 - i * 5}°N
          </text>
        ))}
      </g>

      {/* rhumb lines from the rose */}
      <g stroke="var(--rule)" strokeWidth="0.6" opacity="0.8">
        {Array.from({ length: 32 }, (_, i) => {
          const a = i * 11.25 * DEG;
          return (
            <line
              key={i}
              x1={ROSE[0]}
              y1={ROSE[1]}
              x2={ROSE[0] + Math.sin(a) * 2200}
              y2={ROSE[1] - Math.cos(a) * 2200}
            />
          );
        })}
      </g>

      {/* land, each with two offshore depth contours */}
      {ISLANDS.map((i) => (
        <g key={i.seed} data-coast>
          <path d={coastline(i.c, i.r, i.seed, 1.32)} stroke="var(--rule)" strokeWidth="0.8" strokeDasharray="2 5" />
          <path d={coastline(i.c, i.r, i.seed, 1.15)} stroke="var(--rule-strong)" strokeWidth="0.8" />
          <path d={coastline(i.c, i.r, i.seed)} fill="var(--base-2)" stroke="var(--ink-dim)" strokeWidth="1.3" />
        </g>
      ))}

      {/* soundings, in fathoms */}
      <g fill="var(--ink-faint)" fontFamily="var(--font-mono)" fontSize="10" opacity="0.7">
        {SOUNDINGS.map((s, i) => (
          <text key={i} x={s.x} y={s.y}>
            {s.d}
          </text>
        ))}
      </g>

      <CompassRose />

      {/* the voyage route, revealed by its mask */}
      <path d={routePath} stroke="var(--accent)" strokeWidth="2.6" strokeDasharray="12 10" mask="url(#route-reveal)" />

      {PORTS.map((p, i) => (
        <g key={p.label} data-port transform={`translate(${p.at[0]} ${p.at[1]})`}>
          <rect x={-7} y={-7} width={14} height={14} transform="rotate(45)" fill="var(--base)" stroke="var(--accent)" strokeWidth="2" />
          <rect x={-2.5} y={-2.5} width={5} height={5} transform="rotate(45)" fill="var(--accent)" />
          <text x={p.left ? -14 : 14} y={-14} textAnchor={p.left ? "end" : "start"} fontFamily="var(--font-mono)" fontSize="13" letterSpacing="3">
            <tspan fill="var(--accent-ink)">{String(i + 1).padStart(2, "0")}</tspan>
            <tspan fill="var(--ink-dim)" dx="10">
              {p.label.toUpperCase()}
            </tspan>
          </text>
        </g>
      ))}
    </svg>
  );
}
