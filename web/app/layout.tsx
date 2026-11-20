import type { Metadata } from "next";
import { connection } from "next/server";
import type { ReactNode } from "react";
import { AppShell } from "@/components/shell/AppShell";
import "./globals.css";

export const metadata: Metadata = {
  title: "BowlSense",
  description: "Local AI biomechanics lab for cricket bowling",
};

const LOCAL_API = /^http:\/\/(127\.0\.0\.1|localhost):\d{1,5}$/;

/** The API address, read when the page is served (run.py may pick another port). */
function apiUrl(): string {
  const url = (process.env.BOWLSENSE_API_URL ?? "").replace(/\/$/, "");
  return LOCAL_API.test(url) ? url : "http://127.0.0.1:8000";
}

export default async function RootLayout({ children }: { children: ReactNode }) {
  await connection(); // render per request so the API address is read at run time
  return (
    <html lang="en" className="h-full antialiased">
      <head>
        <meta name="bowlsense-api" content={apiUrl()} />
      </head>
      <body className="min-h-full">
        <a href="#main" className="sr-only focus:not-sr-only focus:p-2">
          Skip to content
        </a>
        <AppShell>
          <main id="main" className="w-full flex-1 px-6 py-5">
            {children}
          </main>
          <footer className="border-t border-border px-6 py-2 text-[11px] text-muted">
            Legality is 2D video screening, not the ICC 3D test. Injury risk is a rule-based
            estimate, not medical advice.
          </footer>
        </AppShell>
      </body>
    </html>
  );
}
