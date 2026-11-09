import { useEffect, useLayoutEffect, type RefObject } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { SplitText } from "gsap/SplitText";
import Lenis from "lenis";

gsap.registerPlugin(ScrollTrigger, SplitText);

export { gsap, ScrollTrigger, SplitText };

export const MOTION_OK = "(prefers-reduced-motion: no-preference)";
export const EASE_REVEAL = "power4.out";
export const EASE_WIPE = "expo.inOut";

const useIsoLayoutEffect = typeof window === "undefined" ? useEffect : useLayoutEffect;

/**
 * Runs GSAP setup scoped to a container, only when the viewer allows motion. Everything it
 * creates is reverted on unmount, so content is static and fully visible otherwise.
 */
export function useMotion<T extends Element>(scope: RefObject<T | null>, setup: (root: T) => void, deps: unknown[] = []) {
  useIsoLayoutEffect(() => {
    const root = scope.current;
    if (!root) return;
    const mm = gsap.matchMedia();
    mm.add(MOTION_OK, () => {
      setup(root);
    });
    return () => mm.revert();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

/** Masked line reveal for every [data-reveal] inside root; [data-reveal="now"] plays immediately. */
export function revealLines(root: Element) {
  root.querySelectorAll<HTMLElement>("[data-reveal]").forEach((el) => {
    const immediate = el.dataset.reveal === "now";
    let played = false;
    // autoSplit re-splits once the web fonts arrive and on resize, so line breaks stay correct.
    SplitText.create(el, {
      type: "lines",
      mask: "lines",
      autoSplit: true,
      onSplit(self) {
        if (played) return;
        return gsap.from(self.lines, {
          yPercent: 110,
          duration: 1.1,
          ease: EASE_REVEAL,
          stagger: 0.07,
          delay: immediate ? Number(el.dataset.delay ?? 0) : 0,
          onComplete: () => {
            played = true;
          },
          scrollTrigger: immediate ? undefined : { trigger: el, start: "top 85%", once: true },
        });
      },
    });
  });
  root.querySelectorAll<HTMLElement>("[data-fade]").forEach((el) => {
    gsap.from(el, {
      autoAlpha: 0,
      y: 16,
      duration: 1,
      ease: EASE_REVEAL,
      delay: Number(el.dataset.delay ?? 0),
      scrollTrigger: el.dataset.fade === "now" ? undefined : { trigger: el, start: "top 88%", once: true },
    });
  });
}

/** Lenis smooth scroll for long editorial pages only (never inside the workspace's scroll panes). */
export function useSmoothScroll(enabled = true) {
  useEffect(() => {
    if (!enabled || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const lenis = new Lenis({ duration: 1.1, easing: (t) => 1 - Math.pow(1 - t, 4) });
    lenis.on("scroll", ScrollTrigger.update);
    const tick = (time: number) => lenis.raf(time * 1000);
    gsap.ticker.add(tick);
    gsap.ticker.lagSmoothing(0);
    return () => {
      gsap.ticker.remove(tick);
      lenis.destroy();
    };
  }, [enabled]);
}
