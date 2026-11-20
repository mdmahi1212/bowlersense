/** A quiet section heading: title, optional hint, optional action on the right. */
export function SectionHead({
  title,
  hint,
  aside,
}: {
  title: string;
  hint?: string;
  aside?: React.ReactNode;
}) {
  return (
    <div className="mb-4 flex flex-wrap items-end justify-between gap-2">
      <div>
        <h2 className="text-lg font-semibold">{title}</h2>
        {hint && <p className="text-sm text-muted">{hint}</p>}
      </div>
      {aside}
    </div>
  );
}
