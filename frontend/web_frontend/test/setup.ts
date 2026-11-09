import "@testing-library/jest-dom/vitest";

// jsdom has no matchMedia. Matching nothing means "(prefers-reduced-motion: no-preference)" is
// false, so tests render the static, no-motion version of every page.
if (!window.matchMedia) {
  window.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener() {},
    removeEventListener() {},
    addListener() {},
    removeListener() {},
    dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia;
}
Element.prototype.scrollIntoView ??= function () {};
