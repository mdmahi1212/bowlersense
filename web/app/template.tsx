import type { ReactNode } from "react";

/** Re-mounted on every navigation, so each page fades up into place. */
export default function Template({ children }: { children: ReactNode }) {
  return <div className="page-enter">{children}</div>;
}
