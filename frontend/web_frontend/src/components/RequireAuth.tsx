import { useLayoutEffect, type ReactNode } from "react";
import { Navigate, useLocation } from "react-router";
import { useAuth } from "../lib/auth";
import { conversationStore, useConversations } from "../lib/conversations";

/**
 * Gates the workspace, and points the conversation store at the signed-in user's history before
 * anything inside renders, so one account never sees another's local history.
 */
export function RequireAuth({ children }: { children: ReactNode }) {
  const { session } = useAuth();
  const { userId } = useConversations();
  const location = useLocation();

  useLayoutEffect(() => {
    conversationStore.activate(session?.userId ?? null);
  }, [session?.userId]);

  if (!session) {
    const next = encodeURIComponent(location.pathname + location.search);
    return <Navigate to={`/signin?next=${next}`} replace />;
  }
  if (userId !== session.userId) return null;
  return <>{children}</>;
}
