import { createBrowserRouter } from "react-router";
import { AppShell } from "./components/AppShell";
import { SignIn, SignUp } from "./pages/Auth";
import { Documents } from "./pages/Documents";
import { Landing } from "./pages/Landing";
import { NotFound } from "./pages/NotFound";
import { Workspace } from "./pages/Workspace";

/*
 * Page paths deliberately differ from the API's (/register, /login, /query, /ingest), so the app
 * and the API can later sit behind one origin without a reverse proxy rule colliding.
 */
export const routes = [
  { path: "/", element: <Landing /> },
  { path: "/signin", element: <SignIn /> },
  { path: "/signup", element: <SignUp /> },
  {
    element: <AppShell />,
    children: [
      { path: "/workspace", element: <Workspace /> },
      { path: "/workspace/:conversationId", element: <Workspace /> },
      { path: "/documents", element: <Documents /> },
    ],
  },
  { path: "*", element: <NotFound /> },
];

export const router = createBrowserRouter(routes);
