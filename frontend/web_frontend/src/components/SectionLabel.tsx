/** `01 — STORY / ストーリー`: index, English label, Japanese label. */
export function SectionLabel({ index, en, ja, className = "" }: { index: string; en: string; ja: string; className?: string }) {
  return (
    <p className={`label flex items-center gap-3 ${className}`}>
      <span className="text-accent">{index}</span>
      <span className="h-px w-8 bg-rule-strong" aria-hidden="true" />
      <span className="text-ink">{en}</span>
      <span className="text-ink-faint">/</span>
      <span lang="ja" className="tracking-[0.1em]">
        {ja}
      </span>
    </p>
  );
}
