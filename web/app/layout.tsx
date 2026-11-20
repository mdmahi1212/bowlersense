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
    <html lang="en" className="h-full antialiased" suppressHydrationWarning>
      <head>
        <meta name="bowlsense-api" content={apiUrl()} />
        <script
          // apply the saved theme before the first paint (no flash of the wrong theme)
          dangerouslySetInnerHTML={{
            __html:
              "try{var t=localStorage.getItem('bowlsense-theme');if(t)document.documentElement.dataset.theme=t}catch(e){}",
          }}
        />
      </head>
      <body className="min-h-full">
        <a href="#main" className="sr-only focus:not-sr-only focus:p-2">
          Skip to content
        </a>
        <AppShell>
          <main id="main" className="mx-auto w-full max-w-[1400px] flex-1 px-8 pb-16 pt-6">
            {children}
          </main>
          <footer className="border-t border-border">
            <p className="mx-auto max-w-[1400px] px-8 py-4 text-xs text-muted">
              Everything runs on this computer. Legality is 2D video screening, not the ICC 3D test.
              Injury risk is a rule-based estimate, not medical advice.
            </p>
          </footer>
        </AppShell>
      </body>
    </html>
  );
}
