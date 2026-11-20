"use client";

import { Button } from "@/components/ui/Button";
import { ErrorBox } from "@/components/ui/States";

export default function ErrorPage({ error, reset }: { error: Error; reset: () => void }) {
  return <ErrorBox error={error.message} action={<Button onClick={reset}>Try again</Button>} />;
}
