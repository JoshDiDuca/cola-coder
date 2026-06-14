interface SparklineProps {
  points: number[];
  width?: number;
  height?: number;
  stroke?: string;
}

// Tiny dependency-free SVG line chart. Scales the y-axis to the min/max of the
// provided points; falls back to a flat midline for empty / single-point /
// all-equal series so the polyline is always well-defined.
export default function Sparkline(props: SparklineProps): JSX.Element {
  const { points, width = 320, height = 64, stroke = '#4f9cf9' } = props;
  const pad = 2;
  const innerW = Math.max(1, width - pad * 2);
  const innerH = Math.max(1, height - pad * 2);

  let path = '';
  if (points.length === 1) {
    const y = pad + innerH / 2;
    path = `${pad},${y} ${pad + innerW},${y}`;
  } else if (points.length > 1) {
    let min = points[0];
    let max = points[0];
    for (const p of points) {
      if (p < min) min = p;
      if (p > max) max = p;
    }
    const span = max - min;
    const n = points.length - 1;
    path = points
      .map((p, i) => {
        const x = pad + (i / n) * innerW;
        // Higher value sits higher on screen (smaller y); flat line when span === 0.
        const y = span === 0 ? pad + innerH / 2 : pad + (1 - (p - min) / span) * innerH;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      role="img"
    >
      {path && (
        <polyline
          points={path}
          fill="none"
          stroke={stroke}
          strokeWidth={1.5}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      )}
    </svg>
  );
}
