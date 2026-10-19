import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { top3, useStore } from '../store.jsx'
import { DEMO_NOTE, STATIC_DEMO, fmtDuration } from '../lib.js'

const W = 1000, H = 660, PAD = 34
const toMap = ([x, y]) => [PAD + x * (W - 2 * PAD), PAD + (1 - y) * (H - 2 * PAD)]

/** Loads one document's map once per document. */
function useSpace(doc, live) {
  const [spaces, setSpaces] = useState({})
  const [error, setError] = useState(null)
  const id = doc?.id
  useEffect(() => {
    if (!id || !live || doc.status !== 'ready' || spaces[id]) return
    setError(null)
    fetch(`api/documents/${id}/space`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((s) => setSpaces((all) => ({ ...all, [id]: s })))
      .catch((e) => setError(e.message))
  }, [id, live, doc?.status, spaces])
  return [spaces[id], error]
}

function Legend({ mode, topics, focusTopic, setFocusTopic }) {
  if (mode !== 'topic') {
    return (
      <div className="ramp-legend">
        <span>{mode === 'position' ? 'start of document' : 'less similar'}</span>
        <i className="ramp" aria-hidden="true" />
        <span>{mode === 'position' ? 'end' : 'more similar to the question'}</span>
      </div>
    )
  }
  return (
    <ul className="topic-legend">
      {topics.map((t) => (
        <li key={t.id}>
          <button
            className={focusTopic === t.id ? 'on' : ''}
            onMouseEnter={() => setFocusTopic(t.id)}
            onMouseLeave={() => setFocusTopic(null)}
            onFocus={() => setFocusTopic(t.id)}
            onBlur={() => setFocusTopic(null)}
          >
            <i className={`sw tp-${t.id}`} aria-hidden="true" />
            <span className="tl-name">{t.name}</span>
            <span className="tl-n">{t.size}</span>
          </button>
        </li>
      ))}
    </ul>
  )
}

export default function Space() {
  const { activeDoc, server } = useStore()
  const live = server.state === 'live'
  const [space, error] = useSpace(activeDoc, live)
  const [proj, setProj] = useState('tsne')
  const [mode, setMode] = useState('topic')
  const [view, setView] = useState({ k: 1, x: 0, y: 0 })
  const [hover, setHover] = useState(null)       // { i, sx, sy }
  const [selected, setSelected] = useState(null) // chunk id
  const [detail, setDetail] = useState(null)
  const [focusTopic, setFocusTopic] = useState(null)
  const [qText, setQText] = useState('')
  const [placed, setPlaced] = useState(null)     // { query, pca, tsne, sims, candidates, timings }
  const [busy, setBusy] = useState(false)
  const [qError, setQError] = useState(null)
  const svg = useRef(null)
  const drag = useRef(null)

  // Reset the view whenever the document changes.
  useEffect(() => { setSelected(null); setPlaced(null); setView({ k: 1, x: 0, y: 0 }) }, [activeDoc?.id])

  const pts = useMemo(() => space?.points.map((p) => ({ ...p, xy: toMap(p[proj]) })) ?? [], [space, proj])
  const n = pts.length
  const centroids = useMemo(() => {
    if (!space) return []
    return space.topics.map((t) => {
      const mine = pts.filter((p) => p.topic === t.id)
      const med = (a) => a.sort((u, v) => u - v)[Math.floor(a.length / 2)]
      return { ...t, xy: [med(mine.map((p) => p.xy[0])), med(mine.map((p) => p.xy[1]))] }
    })
  }, [space, pts])

  // Pointer → map coordinates, accounting for zoom and pan.
  const toLocal = useCallback((e) => {
    const r = svg.current.getBoundingClientRect()
    const s = r.width / W
    const sx = (e.clientX - r.left) / s, sy = (e.clientY - r.top) / s
    return { mx: (sx - view.x) / view.k, my: (sy - view.y) / view.k, sx, sy, s, rx: e.clientX - r.left, ry: e.clientY - r.top }
  }, [view])

  const nearest = useCallback((mx, my, s) => {
    const limit = 14 / (s * view.k)
    let best = null, bd = limit * limit
    for (let i = 0; i < pts.length; i++) {
      const dx = pts[i].xy[0] - mx, dy = pts[i].xy[1] - my, d = dx * dx + dy * dy
      if (d < bd) { bd = d; best = i }
    }
    return best
  }, [pts, view.k])

  // Keep the map covering the viewport, so it can never be panned out of sight.
  const clampView = ({ k, x, y }) => ({
    k, x: Math.min(0, Math.max(W - W * k, x)), y: Math.min(0, Math.max(H - H * k, y)),
  })
  const zoomBy = (f, cx = W / 2, cy = H / 2) => setView((v) => {
    const k = Math.min(14, Math.max(1, v.k * f))
    return clampView({ k, x: cx - (cx - v.x) * (k / v.k), y: cy - (cy - v.y) * (k / v.k) })
  })

  // Active pointers by id; two at once on a touch screen means pinch-zoom.
  const pointers = useRef(new Map())
  const pinch = useRef(null)
  const pinchState = () => {
    const [a, b] = [...pointers.current.values()]
    return { dist: Math.hypot(a.x - b.x, a.y - b.y), mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2 }
  }

  const onMove = (e) => {
    if (!svg.current || !n) return
    const L = toLocal(e)
    if (pointers.current.has(e.pointerId)) pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY })
    if (pinch.current && pointers.current.size === 2) {
      const p = pinchState(), last = pinch.current
      const r = svg.current.getBoundingClientRect(), s = r.width / W
      const cx = (p.mx - r.left) / s, cy = (p.my - r.top) / s
      const f = p.dist / Math.max(1, last.dist)
      setView((v) => {
        const k = Math.min(14, Math.max(1, v.k * f))
        return clampView({ k, x: cx - (cx - v.x) * (k / v.k) + (p.mx - last.mx) / s, y: cy - (cy - v.y) * (k / v.k) + (p.my - last.my) / s })
      })
      pinch.current = p
      return
    }
    if (drag.current) {
      const d = drag.current
      if (Math.hypot(e.clientX - d.cx, e.clientY - d.cy) > 3) d.moved = true
      if (d.moved) setHover(null)
      setView((v) => clampView({ ...v, x: d.x + (e.clientX - d.cx) / L.s, y: d.y + (e.clientY - d.cy) / L.s }))
      return
    }
    const i = nearest(L.mx, L.my, L.s)
    setHover(i == null ? null : { i, rx: L.rx, ry: L.ry })
  }
  const onDown = (e) => {
    pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY })
    e.currentTarget.setPointerCapture?.(e.pointerId)
    if (pointers.current.size === 2) {
      drag.current = null
      pinch.current = pinchState()
      setHover(null)
      return
    }
    drag.current = { cx: e.clientX, cy: e.clientY, x: view.x, y: view.y, moved: false }
  }
  const onUp = (e) => {
    pointers.current.delete(e.pointerId)
    if (pinch.current) {   // lifting one finger of a pinch neither selects nor starts a drag
      if (pointers.current.size === 0) pinch.current = null
      return
    }
    const d = drag.current
    drag.current = null
    if (d && !d.moved) {
      const L = toLocal(e)
      const i = nearest(L.mx, L.my, L.s)
      setSelected(i == null ? null : pts[i].chunk)
    }
  }
  const onCancel = (e) => {
    pointers.current.delete(e.pointerId)
    if (pointers.current.size === 0) pinch.current = null
    drag.current = null
  }
  const onWheel = (e) => {
    if (!e.deltaY) return   // horizontal swipes are not zoom
    // Normalise line/page units (Firefox) to pixels and zoom smoothly with the delta, so trackpads
    // (many small events) and mouse wheels (few large ones) feel alike. Pinch gestures arrive as ctrl+wheel.
    const dy = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? H : 1)
    const f = Math.min(2, Math.max(0.5, Math.exp(-dy * (e.ctrlKey ? 0.01 : 0.002))))
    const L = toLocal(e)
    setHover(null)
    zoomBy(f, L.sx, L.sy)
  }
  // Wheel zoom must not scroll the page, which needs a non-passive listener.
  useEffect(() => {
    const el = svg.current
    if (!el) return
    const stop = (e) => e.preventDefault()
    el.addEventListener('wheel', stop, { passive: false })
    return () => el.removeEventListener('wheel', stop)
  }, [space])

  // Full text and neighbours of the selected chunk.
  useEffect(() => {
    if (selected == null || !activeDoc) { setDetail(null); return }
    let alive = true
    fetch(`api/documents/${activeDoc.id}/chunks/${selected}/neighbours`)
      .then((r) => r.json())
      .then((d) => alive && setDetail(d))
      .catch(() => alive && setDetail(null))
    return () => { alive = false }
  }, [selected, activeDoc])

  const place = async (e) => {
    e.preventDefault()
    const query = qText.trim()
    if (query.length < 3) return
    setBusy(true)
    setQError(null)
    try {
      const r = await fetch(`api/documents/${activeDoc.id}/space/locate`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query }),
      })
      const body = await r.json()
      if (!r.ok) throw new Error(body.detail ?? `HTTP ${r.status}`)
      setPlaced({ query, ...body })
      setMode('similarity')
      setSelected(null)
    } catch (err) {
      setQError(err.message)
    } finally {
      setBusy(false)
    }
  }

  if (!activeDoc) return null

  // Colour and emphasis for each point.
  const simRange = placed ? [Math.min(...placed.sims), Math.max(...placed.sims)] : [0, 1]
  const biTop = new Set(placed ? placed.candidates.map((c) => c.chunk) : [])
  const ceTop = placed ? top3(placed.candidates, 'ceRank') : []
  const bi3 = placed ? top3(placed.candidates, 'biRank') : []
  const ceSet = new Set(ceTop.map((c) => c.chunk))
  const neighbourSet = new Set(detail?.neighbours.map((x) => x.chunk) ?? [])
  const pointStyle = (p) => {
    if (mode === 'topic') return null
    const t = mode === 'position'
      ? p.chunk / Math.max(1, n - 1)
      : (placed.sims[p.chunk] - simRange[0]) / Math.max(1e-6, simRange[1] - simRange[0])
    return { fillOpacity: mode === 'position' ? 0.15 + 0.85 * t : 0.06 + 0.94 * t ** 2.2 }
  }
  const qxy = placed ? toMap(placed[proj]) : null
  const at = (chunk) => pts[chunk]?.xy
  const r = 4 / view.k
  const sel = selected != null ? pts[selected] : null
  const hp = hover ? pts[hover.i] : null
  const topicName = (id) => space?.topics[id]?.name

  return (
    <section className="part wrap space-page">
      <div className="part-head">
        <h2>Chunk space</h2>
        <p className="doc-line">
          Mapping <b>{activeDoc.title}</b>{activeDoc.chunks != null && <> · {activeDoc.chunks} chunks</>} ·{' '}
          <Link to="/">switch or add a PDF</Link>
        </p>
        <p>
          Every dot is one chunk. The bi-encoder turns each chunk into 384 numbers; this map squeezes those into two
          dimensions, so chunks about similar things sit close together. Place a question to see which chunks each
          pipeline picks.
        </p>
      </div>

      {!live && <p className="offline sheet sheet-pad">{STATIC_DEMO ? DEMO_NOTE : 'Start the server to map this document.'}</p>}
      {live && activeDoc.status !== 'ready' && <p className="offline sheet sheet-pad">This document is still being indexed.</p>}
      {error && <p className="red-note">Could not build the map ({error}).</p>}

      <div className="space-toolbar">
        <div className="tabs" role="group" aria-label="Projection">
          <button aria-pressed={proj === 'tsne'} onClick={() => setProj('tsne')}>Meaning map · t-SNE</button>
          <button aria-pressed={proj === 'pca'} onClick={() => setProj('pca')}>Linear view · PCA</button>
        </div>
        <div className="tabs" role="group" aria-label="Colour by">
          <button aria-pressed={mode === 'topic'} onClick={() => setMode('topic')}>Topic</button>
          <button aria-pressed={mode === 'position'} onClick={() => setMode('position')}>Position</button>
          <button aria-pressed={mode === 'similarity'} disabled={!placed} onClick={() => setMode('similarity')}>Similarity</button>
        </div>
        <form className="trace-form space-q" onSubmit={place}>
          <input value={qText} onChange={(e) => setQText(e.target.value)} placeholder="Place a question on the map…"
            disabled={!space} aria-label="Question to place on the map" />
          <button className="btn hot" disabled={!space || busy || qText.trim().length < 3}>{busy ? 'Placing…' : 'Place'}</button>
          {placed && <button type="button" className="btn ghost" onClick={() => { setPlaced(null); setMode('topic') }}>Clear</button>}
        </form>
      </div>
      {qError && <p className="red-note small">{qError}</p>}

      <div className="space-grid">
        <div className="sheet space-map">
          {!space ? (
            <div className="space-loading">
              {live && activeDoc.status === 'ready' && !error ? 'Building the map: embedding projection and topic groups (a few seconds the first time)…' : null}
            </div>
          ) : (
            <>
              <svg
                ref={svg}
                viewBox={`0 0 ${W} ${H}`}
                className={`space-svg ${mode} ${focusTopic != null ? 'focusing' : ''}`}
                role="img"
                aria-label={`Map of ${n} chunks in ${space.topics.length} topic groups`}
                onPointerMove={onMove}
                onPointerDown={onDown}
                onPointerUp={onUp}
                onPointerCancel={onCancel}
                onPointerLeave={() => { if (!pointers.current.size) setHover(null) }}
                onWheel={onWheel}
              >
                <g transform={`translate(${view.x} ${view.y}) scale(${view.k})`}>
                  {/* links: query to picks, or selected chunk to its neighbours */}
                  {placed && bi3.map((c) => at(c.chunk) && (
                    <line key={`b${c.chunk}`} className="lnk bi" x1={qxy[0]} y1={qxy[1]} x2={at(c.chunk)[0]} y2={at(c.chunk)[1]} strokeWidth={1.4 / view.k} />
                  ))}
                  {placed && ceTop.map((c) => at(c.chunk) && (
                    <line key={`c${c.chunk}`} className="lnk ce" x1={qxy[0]} y1={qxy[1]} x2={at(c.chunk)[0]} y2={at(c.chunk)[1]} strokeWidth={2 / view.k} />
                  ))}
                  {sel && detail && detail.chunk === selected && detail.neighbours.map((nb) => at(nb.chunk) && (
                    <line key={`n${nb.chunk}`} className="lnk nb" x1={sel.xy[0]} y1={sel.xy[1]} x2={at(nb.chunk)[0]} y2={at(nb.chunk)[1]} strokeWidth={1.2 / view.k} />
                  ))}

                  {pts.map((p) => (
                    <circle
                      key={p.chunk}
                      r={r}
                      className={`pt tp-${p.topic} ${focusTopic != null && p.topic !== focusTopic ? 'dim' : ''} ${neighbourSet.has(p.chunk) ? 'nbr' : ''}`}
                      style={{ transform: `translate(${p.xy[0]}px, ${p.xy[1]}px)`, ...pointStyle(p) }}
                    />
                  ))}

                  {/* rings: bi-encoder top 10 (blue), cross-encoder top 3 (orange) */}
                  {placed && [...biTop].map((c) => at(c) && (
                    <circle key={`rb${c}`} className="ring bi" cx={at(c)[0]} cy={at(c)[1]} r={8 / view.k} strokeWidth={1.6 / view.k} />
                  ))}
                  {placed && ceTop.map((c, i) => at(c.chunk) && (
                    <g key={`rc${c.chunk}`}>
                      <circle className="ring ce" cx={at(c.chunk)[0]} cy={at(c.chunk)[1]} r={12 / view.k} strokeWidth={2.4 / view.k} />
                      <text className="ring-n" x={at(c.chunk)[0] + 14 / view.k} y={at(c.chunk)[1] - 10 / view.k} fontSize={13 / view.k}
                        strokeWidth={3 / view.k}>{i + 1}</text>
                    </g>
                  ))}
                  {sel && <circle className="ring sel" cx={sel.xy[0]} cy={sel.xy[1]} r={10 / view.k} strokeWidth={2.4 / view.k} />}
                  {hp && <circle className="ring hov" cx={hp.xy[0]} cy={hp.xy[1]} r={7 / view.k} strokeWidth={2 / view.k} />}

                  {/* topic names, placed on the map so colour is never the only cue */}
                  {mode === 'topic' && centroids.map((c) => (
                    <text key={c.id} className={`topic-label ${focusTopic != null && focusTopic !== c.id ? 'dim' : ''}`}
                      x={c.xy[0]} y={c.xy[1]} fontSize={13 / view.k} strokeWidth={4 / view.k} textAnchor="middle">
                      {c.name}
                    </text>
                  ))}

                  {placed && (
                    <g transform={`translate(${qxy[0]} ${qxy[1]}) scale(${1 / view.k})`} className="qstar">
                      <path d="M0,-13 L3.8,-4 L13,-4 L5.6,2 L8.2,11.5 L0,6 L-8.2,11.5 L-5.6,2 L-13,-4 L-3.8,-4 Z" />
                    </g>
                  )}
                </g>
              </svg>

              {hp && !drag.current && (
                <div className="space-tip" style={{ left: Math.min(hover.rx + 14, (svg.current?.clientWidth ?? 600) - 290), top: hover.ry + 14 }}>
                  <div className="tip-head">
                    <b>{hp.cite}</b><span>p. {hp.page} · #{hp.chunk}</span>
                  </div>
                  <div className="tip-topic"><i className={`sw tp-${hp.topic}`} />{topicName(hp.topic)}</div>
                  {placed && <div className="tip-sim">similarity {placed.sims[hp.chunk].toFixed(3)}</div>}
                  <p>{hp.preview}…</p>
                </div>
              )}

              <div className="space-zoom" aria-label="Zoom">
                <button className="btn" onClick={() => zoomBy(1.5)} aria-label="Zoom in">+</button>
                <button className="btn" onClick={() => zoomBy(1 / 1.5)} aria-label="Zoom out">−</button>
                <button className="btn" onClick={() => setView({ k: 1, x: 0, y: 0 })}>Reset</button>
              </div>
              <p className="space-foot">
                {proj === 'tsne'
                  ? 't-SNE keeps neighbours together, so topics form islands; the distance between islands means little.'
                  : 'PCA shows the two directions along which the chunks differ most; faithful overall, but crowded.'}
                {' '}Scroll to zoom, drag to pan, click a dot to read it.
              </p>
            </>
          )}
        </div>

        <aside className="space-side">
          {sel && detail && detail.chunk === selected ? (
            <div className="sheet sheet-pad side-card">
              <div className="side-head">
                <span className="sheet-label">Selected chunk #{detail.chunk}</span>
                <button className="btn ghost" onClick={() => setSelected(null)}>Close</button>
              </div>
              <h3>{detail.cite}{!detail.cite.startsWith('p') && <span> · p. {detail.page}</span>}</h3>
              <div className="tip-topic"><i className={`sw tp-${detail.topic}`} />{topicName(detail.topic)}</div>
              <p className="legal side-text">{detail.text}</p>
              <span className="sheet-label">Nearest chunks (cosine, full 384-d space)</span>
              <ul className="nb-list">
                {detail.neighbours.map((nb) => (
                  <li key={nb.chunk}>
                    <button onClick={() => setSelected(nb.chunk)}>
                      <span className="nb-top"><i className={`sw tp-${nb.topic}`} /><b>{nb.cite}</b><span className="mono">{nb.sim.toFixed(3)}</span></span>
                      <span className="meter"><i style={{ width: `${Math.max(4, nb.sim * 100)}%` }} /></span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ) : placed ? (
            <div className="sheet sheet-pad side-card">
              <div className="side-head">
                <span className="sheet-label">Where the question landed</span>
              </div>
              <p className="side-q">{placed.query}</p>
              {proj === 'tsne' && (
                <p className="side-note">Approximate position: t-SNE cannot place new points, so the star sits among its 5 nearest chunks. Switch to PCA for an exact position.</p>
              )}
              <div className="pick-group">
                <span className="pick-k ce">Cross-encoder top 3 · what the LLM reads</span>
                {ceTop.map((c, i) => (
                  <button key={c.chunk} className="pick ce" onClick={() => setSelected(c.chunk)}>
                    <b>{i + 1}</b><span>{c.cite}</span><small>bi #{c.biRank}</small>
                  </button>
                ))}
              </div>
              <div className="pick-group">
                <span className="pick-k bi">Bi-encoder top 10 · nearest by cosine</span>
                {[...placed.candidates].sort((a, b) => a.biRank - b.biRank).map((c) => (
                  <button key={c.chunk} className={`pick bi ${ceSet.has(c.chunk) ? 'kept' : ''}`} onClick={() => setSelected(c.chunk)}>
                    <b>{c.biRank}</b><span>{c.cite}</span><small>cross #{c.ceRank}</small>
                  </button>
                ))}
              </div>
              <p className="side-note">
                Search {fmtDuration(placed.timings.embedMs + placed.timings.searchMs)} · re-rank {fmtDuration(placed.timings.rerankMs)}
              </p>
            </div>
          ) : (
            <div className="sheet sheet-pad side-card">
              <span className="sheet-label">{mode === 'topic' ? `${space?.topics.length ?? '…'} topic groups` : 'Colour scale'}</span>
              {space && <Legend mode={mode} topics={space.topics} focusTopic={focusTopic} setFocusTopic={setFocusTopic} />}
              <p className="side-note">
                Topics are found by k-means on the full 384-number vectors and named by their most distinctive words.
                Hover a topic to pick it out on the map.
              </p>
            </div>
          )}
        </aside>
      </div>
    </section>
  )
}
