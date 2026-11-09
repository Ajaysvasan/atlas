import { useRef } from "react";
import { gsap, useMotion } from "../lib/motion";

/**
 * Placeholder key visual: an orthographic line drawing of a globe, generated, not traced. The prime
 * meridian is drawn in the accent so it lines up with the page's meridian. Scrolling turns the
 * globe a few degrees; with reduced motion it stays still.
 *
 * To use real art later, render it behind this component, or replace it; the hero expects any
 * full-bleed 2560×1440 image with the subject right of centre.
 */
const R = 300;
const MERIDIANS = Array.from({ length: 12 }, (_, i) => i * 15); // 0..165°, each drawn as a full ellipse
const PARALLELS = [-60, -40, -20, 0, 20, 40, 60];
const POINTS: Array<[number, number, string]> = [
  [52, -18, "PG.MVCC"],
  [31, 34, "SQLITE.WAL"],
  [-14, 62, "CASS.SSTABLE"],
  [-41, -47, "UPLOAD.0007"],
];

function rad(d: number) {
  return (d * Math.PI) / 180;
}

/** A meridian at longitude λ (relative to the viewer) projects to an ellipse of horizontal radius R·|sin λ|. */
function meridianRx(lon: number, rot: number) {
  return Math.abs(R * Math.sin(rad(lon + rot)));
}

function project(lat: number, lon: number, rot: number) {
  const l = rad(lon + rot);
  const p = rad(lat);
  return { x: R * Math.cos(p) * Math.sin(l), y: -R * Math.sin(p), front: Math.cos(l) * Math.cos(p) > 0 };
}

export function Globe({ className = "" }: { className?: string }) {
  const ref = useRef<SVGSVGElement>(null);
  const initialRot = 18;

  useMotion(ref, (svg) => {
    const ellipses = svg.querySelectorAll<SVGEllipseElement>("[data-lon]");
    const points = svg.querySelectorAll<SVGGElement>("[data-point]");
    const state = { rot: initialRot };
    const draw = () => {
      ellipses.forEach((e) => e.setAttribute("rx", meridianRx(Number(e.dataset.lon), state.rot).toFixed(2)));
      points.forEach((g) => {
        const [lat, lon] = (g.dataset.point ?? "0,0").split(",").map(Number);
        const p = project(lat, lon, state.rot);
        g.setAttribute("transform", `translate(${p.x.toFixed(1)} ${p.y.toFixed(1)})`);
        g.style.opacity = p.front ? "1" : "0";
      });
    };
    gsap.from(svg.querySelectorAll("[data-draw]"), {
      strokeDashoffset: 1,
      duration: 2.2,
      ease: "power3.inOut",
      stagger: 0.03,
    });
    gsap.to(state, {
      rot: initialRot + 70,
      ease: "none",
      onUpdate: draw,
      scrollTrigger: { trigger: document.body, start: "top top", end: "bottom bottom", scrub: 1.2 },
    });
  });

  return (
    <svg
      ref={ref}
      viewBox="-360 -360 720 720"
      className={className}
      role="img"
      aria-label="Line drawing of a globe"
      fill="none"
    >
      <circle data-draw r={R} stroke="var(--rule-strong)" strokeWidth="1" pathLength={1} strokeDasharray="1" />
      {PARALLELS.map((lat) => {
        const y = -R * Math.sin(rad(lat));
        const half = R * Math.cos(rad(lat));
        return (
          <line
            key={lat}
            data-draw
            x1={-half}
            x2={half}
            y1={y}
            y2={y}
            stroke="var(--rule)"
            strokeWidth="1"
            pathLength={1}
            strokeDasharray="1"
          />
        );
      })}
      {MERIDIANS.map((lon) => (
        <ellipse
          key={lon}
          data-lon={lon}
          data-draw
          rx={meridianRx(lon, initialRot)}
          ry={R}
          stroke="var(--rule)"
          strokeWidth="1"
          pathLength={1}
          strokeDasharray="1"
        />
      ))}
      {/* the prime meridian: fixed to the viewer, in the accent */}
      <line y1={-R - 40} y2={R + 40} stroke="var(--accent)" strokeWidth="1.25" />
      {[-R - 40, -R / 2, 0, R / 2, R + 40].map((y) => (
        <line key={y} x1={0} x2={10} y1={y} y2={y} stroke="var(--accent)" strokeWidth="1.25" />
      ))}
      {POINTS.map(([lat, lon, label]) => {
        const p = project(lat, lon, initialRot);
        return (
          <g
            key={label}
            data-point={`${lat},${lon}`}
            transform={`translate(${p.x.toFixed(1)} ${p.y.toFixed(1)})`}
            style={{ opacity: p.front ? 1 : 0, transition: "opacity 400ms" }}
          >
            <rect x={-2.5} y={-2.5} width={5} height={5} fill="var(--ink)" />
            <text x={10} y={4} fill="var(--ink-dim)" fontFamily="var(--font-mono)" fontSize="11" letterSpacing="2">
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
