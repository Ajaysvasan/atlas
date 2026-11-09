# Atlas web frontend

The web app for Atlas, built with React and Tailwind CSS. Users create an account, upload
private documents, ask questions, and come back to earlier conversations. It talks only to
the Java REST API ([`backend/atlas`](../../backend/atlas/README.md)), never to the Python
pipeline directly.

[← Project README](../../README.md) · [Java API](../../backend/atlas/README.md) ·
[Python pipeline](../../backend/app/README.md) · [REST API reference](../../backend/BACKEND.md#rest-api)

## Stack

React 19, React Router 8, Tailwind CSS 4 (through `@tailwindcss/vite`), TypeScript, and Vite.
Motion uses GSAP (ScrollTrigger, SplitText) and Lenis. Tests use Vitest, jsdom and Testing
Library. There is no state library: conversations live in a small external store read with
`useSyncExternalStore`.

## Setup

Requires Node.js 20.19+ or 22.12+.

```bash
npm install
cp .env.example .env
```

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `http://localhost:8080` | Where the Java API listens. Restart `npm run dev` after changing it. |

The API's CORS list must include this app's address. The default
(`http://localhost:5173`, `http://localhost:3000`) covers `npm run dev`. Set
`ATLAS_ALLOWED_ORIGINS` on the Java side for anything else.

## Scripts

```bash
npm run dev         # dev server on http://localhost:5173
npm run build       # type check, then a production build into dist/
npm run preview     # serve the production build
npm test            # Vitest, once
npm run typecheck   # TypeScript only
```

For working pages, the Java API and the Python pipeline must both be running; see
[Running](../../README.md#running) in the project README. Without them, the landing page
still works and the app reports the API as unreachable.

## Pages

| Route | Page |
|---|---|
| `/` | Landing page: what Atlas does and how it works |
| `/signin`, `/signup` | Sign in and registration. Validation matches the API's rules. |
| `/workspace` | Ask a question to start a new conversation. The history rail lists past conversations from the server. |
| `/workspace/:conversationId` | Continue a conversation. Its earlier questions and answers load from the server. |
| `/documents` | Upload documents (PDF, DOCX, PPTX, XLSX, TXT, MD, HTML, JSON, CSV), with progress and per-file results |
| anything else | Not found |

`/workspace` and `/documents` require sign-in. The header shows whether the API is reachable.

## How it talks to the API

All calls go through [`src/lib/api.ts`](src/lib/api.ts):

- **Endpoints:** `/register`, `/login`, `/query`, `/ingest`, `/conversations`,
  `/conversations/{id}/exchanges` and `/actuator/health`.
- **Headers:** every request sends an `X-Request-Id` (`web-<uuid>`).
- **Errors:** they come back as `ApiError`, carrying the server's `code`, field errors and
  `Retry-After`.
- **Uploads:** they use `XMLHttpRequest` so progress can be shown.
- **Queries:** they can take minutes on a CPU, so the composer shows elapsed time and allows
  cancelling.

Auth and history are handled like this:

- **Session:** [`src/lib/auth.tsx`](src/lib/auth.tsx) keeps the JWT in `sessionStorage`. It
  survives a reload but not closing the tab. The app signs out on expiry or on any 401 from
  an authenticated call.
- **History:** [`src/lib/conversations.ts`](src/lib/conversations.ts) reads conversation
  history from the server, paged with "Load more".
- **Cached answer details:** each answer's status, timing, retry count and warnings aren't
  stored on the server, so they're cached per user in `localStorage`.

## Layout

```text
src/
  main.tsx, router.tsx        entry and routes
  lib/                        api client, auth, conversation store, motion helpers, formatting
  components/                 header and menu, app shell, meridian, globe, answer renderer, ...
  pages/                      Landing, Auth, Workspace, Documents, NotFound
  styles/index.css            design tokens and Tailwind theme
test/                         Vitest suites (api client, conversation store, routing, components)
```

## Design

The look is cold, sci-fi editorial: a near-black base (`#05070A`), pale ink (`#D7E2EA`) and
one orange accent (`#FF7A00`). The fonts are Syne for display, Zen Kaku Gothic New for body
text and JetBrains Mono for labels, with film grain over the page. The recurring element is
the meridian, a thin orange line with coordinate ticks. All colours come from the tokens in
[`src/styles/index.css`](src/styles/index.css). Animation runs only when the system allows
motion (`prefers-reduced-motion`). Otherwise everything is shown without animation.
