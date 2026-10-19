import { useEffect, useState } from 'react'
import Flap from './Flap.jsx'

const fit = (s, n) => (s.length > n ? `${s.slice(0, n - 1)}…` : s)

/**
 * A departure board that re-sorts itself: the ten candidates for one real question, flipping between
 * bi-encoder order and cross-encoder order every few seconds. The top three rows are what the LLM reads.
 */
export default function ReRankBoard({ query }) {
  const [mode, setMode] = useState('bi')
  const [paused, setPaused] = useState(false)
  useEffect(() => {
    if (paused) return
    const t = setInterval(() => setMode((m) => (m === 'bi' ? 'ce' : 'bi')), 3600)
    return () => clearInterval(t)
  }, [paused])

  const key = mode === 'bi' ? 'biRank' : 'ceRank'
  const rows = [...query.candidates].sort((a, b) => a[key] - b[key])
  const score = (c) => (mode === 'bi' ? c.biScore.toFixed(3) : c.ceScore.toFixed(2))

  return (
    <figure className={`board-panel ${mode}`} onMouseEnter={() => setPaused(true)} onMouseLeave={() => setPaused(false)}>
      <div className="board-head">
        <Flap text={mode === 'bi' ? 'Bi-encoder order' : 'Cross-encoder order'} width={19} className="board-title" />
        <span className="board-q">“{query.query}”</span>
      </div>
      <div className="board-scroll">
        <table className="board-table">
          <thead>
            <tr><th>Rank</th><th>Source</th><th>Score</th><th>Status</th></tr>
          </thead>
          <tbody>
            {rows.map((c, i) => (
              <tr key={i} className={i < 3 ? 'boarding' : ''}>
                <td><Flap text={String(i + 1).padStart(2, ' ')} width={2} /></td>
                <td><Flap text={fit(c.cite, 14)} width={14} /></td>
                <td><Flap text={score(c)} width={6} /></td>
                <td><Flap text={i < 3 ? (c.gold ? 'LLM ★' : 'LLM') : c.gold ? 'ANSWER ★' : ''} width={8} className={i < 3 ? 'go' : ''} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <figcaption>
        <span>Same ten chunks, two orderings. The top three rows go to the LLM; ★ is the passage with the answer.</span>
        <button className="board-toggle" onClick={() => setMode((m) => (m === 'bi' ? 'ce' : 'bi'))}>
          Flip to {mode === 'bi' ? 'cross-encoder' : 'bi-encoder'}
        </button>
      </figcaption>
    </figure>
  )
}
