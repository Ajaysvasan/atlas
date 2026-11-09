import { describe, expect, it } from "vitest";
import { parseBlocks } from "../src/components/Answer";
import { decodeSubject } from "../src/lib/auth";
import { formatDuration } from "../src/lib/format";
import { validateRegistration } from "../src/pages/Auth";
import { checkFile } from "../src/pages/Documents";

describe("decodeSubject", () => {
  it("reads sub from a base64url JWT payload", () => {
    const payload = btoa(JSON.stringify({ sub: "6f1c-uuid", exp: 1 })).replace(/=+$/, "").replace(/\+/g, "-").replace(/\//g, "_");
    expect(decodeSubject(`h.${payload}.s`)).toBe("6f1c-uuid");
    expect(decodeSubject("garbage")).toBeNull();
  });
});

describe("parseBlocks", () => {
  it("splits paragraphs, lists, headings and fenced code", () => {
    const blocks = parseBlocks("## Title\n\nOne\nline.\n\n- a\n- b\n\n1. x\n\n```sql\nSELECT 1;\n```");
    expect(blocks).toEqual([
      { kind: "h", level: 3, text: "Title" },
      { kind: "p", text: "One line." },
      { kind: "ul", items: ["a", "b"] },
      { kind: "ol", items: ["x"] },
      { kind: "code", text: "SELECT 1;", lang: "sql" },
    ]);
  });
});

describe("validateRegistration", () => {
  it("mirrors the server's rules", () => {
    expect(validateRegistration({ username: "dev", email: "d@e.f", password: "12345678", confirm: "12345678" })).toEqual({});
    const e = validateRegistration({ username: "_x", email: "nope", password: "short", confirm: "other" });
    expect(Object.keys(e).sort()).toEqual(["confirm", "email", "password", "username"]);
  });
});

describe("checkFile", () => {
  it("accepts the allowlist and rejects the rest", () => {
    expect(checkFile(new File(["x"], "notes.md"))).toBeUndefined();
    expect(checkFile(new File(["x"], "Report.PDF"))).toBeUndefined();
    expect(checkFile(new File(["x"], "tool.exe"))).toMatch(/not accepted/);
    expect(checkFile(new File([], "empty.txt"))).toMatch(/empty/);
  });
});

describe("formatDuration", () => {
  it("formats durations and clocks", () => {
    expect(formatDuration(850)).toBe("850 ms");
    expect(formatDuration(12_400)).toBe("12.4 s");
    expect(formatDuration(187_000)).toBe("3 m 07 s");
    expect(formatDuration(187_000, true)).toBe("03:07");
  });
});
