/**
 * The signature move: a 1px accent line with coordinate ticks. It is the landing page's spine, the
 * conversation's timeline, and — with `running` — the indicator that the pipeline is working.
 */
interface MeridianProps {
  className?: string;
  /** Tick labels, evenly spaced from top to bottom. */
  ticks?: string[];
  /** Index of the highlighted tick. */
  active?: number;
  running?: boolean;
}

export function Meridian({ className = "relative", ticks = [], active = -1, running = false }: MeridianProps) {
  // className must carry the positioning (relative, absolute or fixed) and the height.
  return (
    <div aria-hidden="true" className={`pointer-events-none w-px ${className}`}>
      <div className="absolute inset-y-0 left-0 w-px bg-accent/70" />
      {running && (
        <div className="absolute inset-y-0 left-0 w-px overflow-hidden">
          <div className="meridian-scan absolute left-[-1px] top-0 h-24 w-[3px] bg-gradient-to-b from-transparent via-accent to-transparent" />
        </div>
      )}
      {ticks.map((label, i) => {
        const top = ticks.length === 1 ? 0 : (i / (ticks.length - 1)) * 100;
        const on = i === active;
        return (
          <div key={label + i} className="absolute left-0 flex items-center" style={{ top: `${top}%` }}>
            <span className={`h-px transition-all duration-700 ${on ? "w-6 bg-accent" : "w-2.5 bg-accent/70"}`} />
            <span
              className={`ml-2 whitespace-nowrap font-mono text-[10px] tracking-[0.2em] transition-colors duration-700 ${
                on ? "text-accent" : "text-ink-faint"
              }`}
            >
              {label}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/** One tick, for inline use (the conversation timeline places one per turn). */
export function MeridianTick({ label, active = false }: { label: string; active?: boolean }) {
  return (
    <div aria-hidden="true" className="flex items-center gap-2">
      <span className={`h-px ${active ? "w-6 bg-accent" : "w-3 bg-accent/70"}`} />
      <span className={`font-mono text-[10px] tracking-[0.2em] ${active ? "text-accent" : "text-ink-faint"}`}>{label}</span>
    </div>
  );
}
