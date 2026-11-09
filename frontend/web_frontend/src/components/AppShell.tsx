import { Outlet } from "react-router";
import { Header } from "./Header";
import { RequireAuth } from "./RequireAuth";

/** Signed-in chrome: the sticky header over the workspace and documents pages. */
export function AppShell() {
  return (
    <RequireAuth>
      <Header variant="app" />
      <main>
        <Outlet />
      </main>
    </RequireAuth>
  );
}
