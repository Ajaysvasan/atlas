const pad = (n: number) => String(n).padStart(2, "0");

/** `2026.11.09 14:32`, the dotted date format used across the interface. */
export function formatStamp(ms: number): string {
  const d = new Date(ms);
  return `${d.getFullYear()}.${pad(d.getMonth() + 1)}.${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** `850 ms`, `12.4 s`, `3 m 07 s`; `clock` gives `03:07` for a running timer. */
export function formatDuration(ms: number, clock = false): string {
  const total = Math.floor(ms / 1000);
  if (clock) {
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    return h ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
  }
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  const m = Math.floor(total / 60);
  return `${m} m ${pad(total % 60)} s`;
}
