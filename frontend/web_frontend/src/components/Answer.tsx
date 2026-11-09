import { Fragment, type ReactNode } from "react";

/**
 * Renders the model's answer: paragraphs, headings, lists, fenced code, inline `code` and **bold**.
 * It builds React elements and never injects HTML, so nothing in an answer can run as markup.
 */
type Block =
  | { kind: "p"; text: string }
  | { kind: "h"; level: 3 | 4; text: string }
  | { kind: "ul" | "ol"; items: string[] }
  | { kind: "code"; text: string; lang: string };

export function parseBlocks(src: string): Block[] {
  const lines = src.replace(/\r\n?/g, "\n").split("\n");
  const blocks: Block[] = [];
  let para: string[] = [];
  let list: { kind: "ul" | "ol"; items: string[] } | null = null;

  const flushPara = () => {
    if (para.length) blocks.push({ kind: "p", text: para.join(" ") });
    para = [];
  };
  const flushList = () => {
    if (list) blocks.push(list);
    list = null;
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const fence = line.match(/^\s*```\s*([\w+-]*)\s*$/);
    if (fence) {
      flushPara();
      flushList();
      const body: string[] = [];
      i++;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) body.push(lines[i++]);
      blocks.push({ kind: "code", text: body.join("\n"), lang: fence[1] });
      continue;
    }
    const heading = line.match(/^\s{0,3}(#{1,6})\s+(.*)$/);
    if (heading) {
      flushPara();
      flushList();
      blocks.push({ kind: "h", level: heading[1].length <= 2 ? 3 : 4, text: heading[2].replace(/\s*#+\s*$/, "") });
      continue;
    }
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || numbered) {
      flushPara();
      const kind = bullet ? "ul" : "ol";
      if (list && list.kind !== kind) flushList();
      if (!list) list = { kind, items: [] };
      list.items.push((bullet ?? numbered)![1]);
      continue;
    }
    if (!line.trim()) {
      flushPara();
      flushList();
      continue;
    }
    if (list) {
      // a wrapped continuation of the previous item
      const items: string[] = (list as { items: string[] }).items;
      items[items.length - 1] += ` ${line.trim()}`;
      continue;
    }
    para.push(line.trim());
  }
  flushPara();
  flushList();
  return blocks;
}

export function renderInline(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(`[^`]+`|\*\*[^*]+\*\*)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const tok = m[0];
    out.push(tok.startsWith("`") ? <code key={k++}>{tok.slice(1, -1)}</code> : <strong key={k++}>{tok.slice(2, -2)}</strong>);
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function Answer({ text }: { text: string }) {
  const blocks = parseBlocks(text);
  return (
    <div className="prose-atlas text-[16px] leading-[1.85] text-ink/90">
      {blocks.map((b, i) => {
        switch (b.kind) {
          case "p":
            return <p key={i}>{renderInline(b.text)}</p>;
          case "h":
            return b.level === 3 ? <h3 key={i}>{renderInline(b.text)}</h3> : <h4 key={i}>{renderInline(b.text)}</h4>;
          case "ul":
          case "ol": {
            const Tag = b.kind;
            return (
              <Tag key={i}>
                {b.items.map((it, j) => (
                  <li key={j}>{renderInline(it)}</li>
                ))}
              </Tag>
            );
          }
          case "code":
            return (
              <pre key={i} data-lang={b.lang || undefined}>
                <code>{b.text}</code>
              </pre>
            );
          default:
            return <Fragment key={i} />;
        }
      })}
    </div>
  );
}
