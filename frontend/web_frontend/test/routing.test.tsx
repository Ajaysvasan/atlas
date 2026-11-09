import { render, screen } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AuthProvider } from "../src/lib/auth";
import { routes } from "../src/router";

beforeEach(() => {
  sessionStorage.clear();
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: "UP" }), { status: 200 })));
});

function renderAt(path: string) {
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  render(<AuthProvider><RouterProvider router={router} /></AuthProvider>);
  return router;
}

describe("routing", () => {
  it("sends a signed-out visitor from the workspace to sign-in, remembering where they were going", async () => {
    const router = renderAt("/workspace/abc");
    expect(await screen.findByRole("heading", { name: "Welcome back." })).toBeInTheDocument();
    expect(router.state.location.pathname).toBe("/signin");
    expect(router.state.location.search).toBe("?next=%2Fworkspace%2Fabc");
  });

  it("opens the workspace for a signed-in user", async () => {
    sessionStorage.setItem("atlas.session", JSON.stringify({ accessToken: "t", userId: "u1", email: "a@b.c", expiresAt: Date.now() + 3_600_000 }));
    renderAt("/workspace");
    expect(await screen.findByRole("heading", { name: "What do you want to know?" })).toBeInTheDocument();
  });

  it("shows the 404 page for unknown paths", async () => {
    renderAt("/nowhere");
    expect(await screen.findByRole("heading", { name: "No such page." })).toBeInTheDocument();
  });
});
