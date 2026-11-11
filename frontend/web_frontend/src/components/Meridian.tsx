/**
 * The signature move: the voyage route — a dashed vermilion line with diamond port markers. It is the
 * landing page's spine (one port per section), the edge of the sign-in panel, the conversation's log,
 * and — with `running` — the ship's marker sailing down the line while the pipeline works.
 */
interface MeridianProps {
  className?: string;
  /** Port labels, evenly spaced from top to bottom. */
  ticks?: string[];
  /** Index of the current port. */
  active?: number;
  running?: boolean;
}

export function Meridian({ className = "relative", ticks = [], active = -1, running = false }: MeridianProps) {
  // className must carry the positioning (relative, absolute or fixed) and the height.
  return (
    <div aria-hidden="true" className={`pointer-events-none w-px ${className}`}>
      <div className="route-line absolute inset-y-0 left-[-0.5px] w-[2px]" />
      {running && (
        <div className="absolute inset-y-0 left-[-5px] w-[11px] overflow-hidden">
          <div className="route-sail absolute left-[1px] top-0 h-[9px] w-[9px] bg-accent" />
        </div>
      )}
      {ticks.map((label, i) => {
        const top = ticks.length === 1 ? 0 : (i / (ticks.length - 1)) * 100;
        const on = i === active;
        return (
          <div key={label + i} className="absolute left-[-4px] flex -translate-y-1/2 items-center" style={{ top: `${top}%` }}>
            <span className="port" data-on={on} />
            <span
              className={`ml-3 whitespace-nowrap font-mono text-[10px] tracking-[0.2em] transition-colors duration-700 ${
                on ? "text-accent-ink" : "text-ink-faint"
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

/** One port, for inline use (the conversation's log places one per turn). */
export function MeridianTick({ label, active = false }: { label: string; active?: boolean }) {
  return (
    // -ml-[4px] centres the 9px diamond on a route line drawn at the same left edge.
    <div aria-hidden="true" className="-ml-[4px] flex items-center gap-3">
      <span className="port" data-on={active} />
      <span className={`font-mono text-[10px] tracking-[0.2em] ${active ? "text-accent-ink" : "text-ink-faint"}`}>{label}</span>
    </div>
  );
}
