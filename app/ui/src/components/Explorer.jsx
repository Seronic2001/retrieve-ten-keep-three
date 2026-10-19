import { useEffect, useRef, useState } from 'react'
import { CATEGORY, VERDICT, byCe, markSnippets, rankText, sigmoid } from '../lib.js'
import Flap from './Flap.jsx'

function QueryPicker({ queries, qi, onPick }) {
  return (
    <div className="picker" role="group" aria-label="Choose a test question">
      {queries.map((q, i) => (
        <button key={q.query} className="slip" aria-pressed={i === qi} onClick={() => onPick(i)}>
          <span className="top">
            <span className="cat">{CATEGORY[q.category]}</span>
            <span className="shift">{rankText(q.baselineRank)} → {rankText(q.rerankedRank)}</span>
          </span>
          <span className="q">{q.query}</span>
          <span className={`verdict-word ${q.status}`}>{VERDICT[q.status]}</span>
        </button>
      ))}
    </div>
  )
}

function Slope({ query, selected, onSelect }) {
  const W = 300, top = 40, gap = 27, xl = 64, xr = W - 64
  const y = (r) => top + (r - 1) * gap
  const H = y(10) + 22
  const ordered = [...query.candidates].sort((a, b) => a.gold - b.gold) // gold drawn last, on top
  return (
    <div className="sheet sheet-pad slope">
      <span className="sheet-label">Every move at once</span>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Rank changes from bi-encoder to cross-encoder">
        <rect className="zone" x={xl - 16} y={y(1) - 13} width={32} height={gap * 2 + 26} rx={14} />
        <rect className="zone" x={xr - 16} y={y(1) - 13} width={32} height={gap * 2 + 26} rx={14} />
        <text className="t" x={xl} y={16} textAnchor="middle">BI</text>
        <text className="t" x={xr} y={16} textAnchor="middle">CROSS</text>
        {Array.from({ length: 10 }, (_, i) => (
          <g key={i}>
            <text className="n" x={xl - 26} y={y(i + 1) + 3.5} textAnchor="end">{i + 1}</text>
            <text className="n" x={xr + 26} y={y(i + 1) + 3.5}>{i + 1}</text>
          </g>
        ))}
        {ordered.map((c) => {
          const dir = c.ceRank < c.biRank ? 'up' : c.ceRank > c.biRank ? 'down' : ''
          const pts = { x1: xl, y1: y(c.biRank), x2: xr, y2: y(c.ceRank) }
          return (
            <g key={c.chunk} className="hit" onClick={() => onSelect(c.chunk)}>
              <line {...pts} stroke="transparent" strokeWidth={14} />
              {c.gold && <line className="l gold" {...pts} />}
              <line className={`l ${c.gold ? 'gold-ink' : dir} ${selected === c.chunk ? 'sel' : ''}`} {...pts} />
              <circle cx={xl} cy={y(c.biRank)} r={4.5} />
              <circle cx={xr} cy={y(c.ceRank)} r={4.5} />
            </g>
          )
        })}
      </svg>
      <p>Dashed loops = the 3 that reach the LLM. Highlighted line = the answer.</p>
    </div>
  )
}

export default function Explorer({ query, queries, qi, onPick }) {
  const [mode, setMode] = useState('ce')
  const goldOrTop = () => (query.candidates.find((c) => c.gold) ?? byCe(query)[0]).chunk
  const [selected, setSelected] = useState(goldOrTop)
  const timer = useRef(null)
  const firstRender = useRef(true)

  useEffect(() => {
    // Open fully sorted; animate only when the reader picks another question.
    if (firstRender.current) { firstRender.current = false; return }
    setSelected(goldOrTop())
    replay()
    return () => clearTimeout(timer.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query])

  function replay() {
    clearTimeout(timer.current)
    setMode('bi')
    timer.current = setTimeout(() => setMode('ce'), 900)
  }

  const rankOf = (c) => (mode === 'bi' ? c.biRank : c.ceRank)
  const sel = query.candidates.find((c) => c.chunk === selected) ?? query.candidates[0]
  const hasGold = query.candidates.some((c) => c.gold)

  return (
    <section id="replay" className="part wrap">
      <div className="part-head">
        <h2>Watch ten chunks swap places</h2>
        <p>
          Pick a question. The list starts in bi-encoder order and re-sorts by cross-encoder score. Only the
          bracketed top three reach the LLM. Click a row to read the chunk.
        </p>
      </div>

      <QueryPicker queries={queries} qi={qi} onPick={(i) => onPick(i)} />

      <div className="ex-grid">
        <div className="sheet sheet-pad ex-main">
          <div className="ex-toolbar">
            <div className="bigshift">
              <span className="lbl">answer chunk moved</span>
              <b>{rankText(query.baselineRank)}</b>
              <span className="arrow">→</span>
              <b className={`to ${query.status}`}>{rankText(query.rerankedRank)}</b>
            </div>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              <div className="tabs" role="group" aria-label="Order">
                <button aria-pressed={mode === 'bi'} onClick={() => setMode('bi')}>Bi-encoder order</button>
                <button aria-pressed={mode === 'ce'} onClick={() => setMode('ce')}>Cross-encoder order</button>
              </div>
              <button className="btn hot" onClick={replay}>↻ Replay</button>
            </div>
          </div>

          {!hasGold && (
            <p className="miss-note">The answer chunk never made the top 10, so re-ranking has nothing to promote.</p>
          )}

          <div className="ranklist">
            <div className="ctx-bracket"><span>to the LLM</span></div>
            {query.candidates.map((c) => {
              const r = rankOf(c)
              const score = mode === 'bi' ? c.biScore : sigmoid(c.ceScore)
              return (
                <button
                  key={c.chunk}
                  className={`rrow ${c.gold ? 'gold' : ''}`}
                  style={{ transform: `translateY(calc(var(--row) * ${r - 1}))` }}
                  aria-pressed={selected === c.chunk}
                  onClick={() => setSelected(c.chunk)}
                >
                  <span className="rk"><Flap text={String(r).padStart(2, ' ')} width={2} /></span>
                  <span className="body">
                    <span className="cite">{c.cite}<small>p. {c.page} · #{c.chunk}</small></span>
                    <span className="prev"><span>{c.text}</span></span>
                  </span>
                  <span className="sc">
                    {mode === 'bi' ? c.biScore.toFixed(3) : c.ceScore.toFixed(2)}
                    <span className="meter"><i style={{ width: `${Math.max(4, score * 100)}%` }} /></span>
                  </span>
                </button>
              )
            })}
          </div>
        </div>

        <div className="ex-side">
          <Slope query={query} selected={selected} onSelect={setSelected} />
          <article className="sheet sheet-pad chunk">
            <div className="chunk-head">
              <h3>{sel.cite} · p. {sel.page}</h3>
              {sel.gold && <span className="gold-tag">the answer</span>}
            </div>
            <div className="chunk-scores">
              <span>bi <b>#{sel.biRank}</b> ({sel.biScore.toFixed(3)})</span>
              <span>cross <b>#{sel.ceRank}</b> ({sel.ceScore.toFixed(2)})</span>
              <span>chunk <b>#{sel.chunk}</b></span>
            </div>
            <p className="legal">
              {markSnippets(sel.text, query.goldSnippets).map((p, i) =>
                p.hit ? <mark key={i}>{p.text}</mark> : <span key={i}>{p.text}</span>,
              )}
            </p>
          </article>
        </div>
      </div>
    </section>
  )
}
