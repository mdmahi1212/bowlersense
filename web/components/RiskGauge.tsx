import { bandOf, bandStyle } from "@/lib/utils";

/** Semi-circular 0-100 gauge with the band written out (colour is not the only signal). */
export function RiskGauge({ score, label = "Injury risk" }: { score: number; label?: string }) {
  const band = bandOf(score);
  const style = bandStyle[band];
  const angle = Math.PI * (1 - Math.min(100, Math.max(0, score)) / 100);
  const x = 60 + 48 * Math.cos(angle);
  const y = 60 - 48 * Math.sin(angle);
  return (
    <figure
      className="flex flex-col items-center"
      aria-label={`${label}: ${score.toFixed(0)} of 100, ${style.label}`}
    >
      <svg viewBox="0 0 120 70" className="w-40" role="img" aria-hidden="true">
        <path
          d="M12 60 A48 48 0 0 1 108 60"
          fill="none"
          stroke="var(--surface-2)"
          strokeWidth="10"
        />
        <path
          d={`M12 60 A48 48 0 0 1 ${x.toFixed(2)} ${y.toFixed(2)}`}
          fill="none"
          stroke={style.fill}
          strokeWidth="10"
          strokeLinecap="round"
        />
        <text
          x="60"
          y="58"
          textAnchor="middle"
          fontSize="20"
          fontWeight="700"
          fill="var(--foreground)"
        >
          {score.toFixed(0)}
        </text>
      </svg>
      <figcaption className="text-center text-sm">
        <span className={`font-semibold ${style.text}`}>{style.label} risk</span>
        <span className="block text-xs text-muted">{label} (screening, not medical advice)</span>
      </figcaption>
    </figure>
  );
}
