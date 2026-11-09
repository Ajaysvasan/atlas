import { useEffect, useState } from "react";
import { api } from "../lib/api";

/**
 * GET /actuator/health, polled every 30 s while the tab is visible. It reports the Java service
 * only; a down Python pipeline shows up as BACKEND_UNAVAILABLE on the next query.
 */
export function ApiStatus() {
  const [up, setUp] = useState<boolean | null>(null);

  useEffect(() => {
    let controller = new AbortController();
    const check = () => {
      if (document.visibilityState !== "visible") return;
      controller.abort();
      controller = new AbortController();
      void api.health(controller.signal).then(setUp);
    };
    check();
    const id = window.setInterval(check, 30_000);
    document.addEventListener("visibilitychange", check);
    return () => {
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", check);
      controller.abort();
    };
  }, []);

  const text = up === null ? "Checking" : up ? "API online" : "API offline";
  return (
    <span className="label flex items-center gap-2" role="status" aria-live="polite">
      <span
        aria-hidden="true"
        className={`inline-block h-1.5 w-1.5 ${up === null ? "bg-ink-faint" : up ? "bg-ink" : "bg-danger"}`}
      />
      {text}
    </span>
  );
}
