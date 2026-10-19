import { useEffect, useState } from 'react'

/** One split-flap cell: when its character changes it flips over after `delay` ms. */
function FlapCell({ ch, delay }) {
  const [shown, setShown] = useState(ch)
  const [flips, setFlips] = useState(0)
  useEffect(() => {
    if (ch === shown) return
    const t = setTimeout(() => { setShown(ch); setFlips((f) => f + 1) }, delay)
    return () => clearTimeout(t)
  }, [ch, shown, delay])
  return (
    <span className="flap-cell" aria-hidden="true">
      <span key={flips} className={flips ? 'flap-ch flipping' : 'flap-ch'}>{shown === ' ' ? ' ' : shown}</span>
    </span>
  )
}

/** Split-flap text, like a departure board. Cells flip left to right when the text changes. */
export default function Flap({ text, width, className = '', stagger = 22 }) {
  const raw = String(text ?? '').toUpperCase()
  const s = width ? raw.padEnd(width).slice(0, width) : raw
  return (
    <span className={`flap ${className}`} role="text" aria-label={raw.trim()}>
      {[...s].map((ch, i) => <FlapCell key={i} ch={ch} delay={i * stagger} />)}
    </span>
  )
}
