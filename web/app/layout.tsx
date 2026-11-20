import type { Metadata } from "next";
import { connection } from "next/server";
import type { ReactNode } from "react";
import { Nav } from "@/components/Nav";
import "./globals.css";

export const metadata: Metadata = {
  title: "BowlSense",
  description: "Local AI analysis of cricket bowling actions",
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
      <body className="flex min-h-full flex-col">
        <a href="#main" className="sr-only focus:not-sr-only focus:p-2">
          Skip to content
        </a>
        <Nav />
        <main id="main" className="mx-auto w-full max-w-6xl flex-1 px-4 py-6">
          {children}
        </main>
        <footer className="border-t border-border px-4 py-3 text-center text-xs text-muted">
          Runs entirely on this computer. Legality is 2D video screening only; injury risk is a
          rule-based estimate, not medical advice.
        </footer>
      </body>
    </html>
  );
}
