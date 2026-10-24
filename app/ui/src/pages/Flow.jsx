import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { DEFAULT_RERANKER, GDPR_ID, pipelineLabels, top3, useStore } from '../store.jsx'
import PipelinePicker from '../components/PipelinePicker.jsx'
import { DEMO_NOTE, STATIC_DEMO, fmtDuration } from '../lib.js'

// Stage names; the search and re-rank subtitles follow the question's retrieval mode and re-ranker.
const stages = (hybrid, reranker) => [
  ['Chunk & embed', 'Offline, once'],
  ['Search', hybrid ? 'Dense + BM25, fused' : 'Bi-encoder top 10'],
  ['Re-rank', reranker],
  ['Select', 'Each keeps 3'],
  ['Generate', 'Two answers'],
]
const ROW = 40

/* ── part 1: how the PDF is cut ─────────────────────────────── */
function Chunking({ map }) {
  const ex = map.example
  const { chunkWords: size, overlapWords: overlap } = map
  const step = size - overlap
  const span = step * (ex.length - 1) + size
  const [pick, setPick] = useState(Math.min(1, ex.length - 1))

  const words = ex[pick].text.split(' ')
  const head = pick > 0 ? overlap : 0
  const tail = pick < ex.length - 1 ? overlap : 0

  return (
    <div className="sheet sheet-pad chunking">
      <div className="chunk-diagram" role="img" aria-label={`Windows of ${size} words, each starting ${step} words after the last`}>
        {ex.map((c, i) => (
          <button
            key={c.chunk}
            className={`window ${pick === i ? 'on' : ''}`}
            style={{ marginLeft: `${((i * step) / span) * 100}%`, width: `${(size / span) * 100}%` }}
            onClick={() => setPick(i)}
          >
            {i > 0 && <i className="ov" style={{ width: `${(overlap / size) * 100}%` }} />}
            <span>#{c.chunk} · {c.cite}</span>
            {i < ex.length - 1 && <i className="ov end" style={{ width: `${(overlap / size) * 100}%` }} />}
          </button>
        ))}
        <div className="ruler">
          {Array.from({ length: Math.floor(span / 100) + 1 }, (_, k) => (
            <span key={k} style={{ left: `${((k * 100) / span) * 100}%` }}>{k * 100}</span>
          ))}
        </div>
      </div>
      <div className="chunking-copy">
        <p>
          Each chunk is <b>{size} words</b> and the next one starts <b>{step} words</b> later, so neighbours share{' '}
          <b>{overlap} words</b> (striped). A sentence cut at one boundary still appears whole in the neighbouring
          chunk. These are three real chunks from the middle of the document; click a window to read it.
        </p>
        <p className="legal chunk-read">
          {head > 0 && <span className="ov-text">{words.slice(0, head).join(' ')} </span>}
          {words.slice(head, words.length - tail).join(' ')}
          {tail > 0 && <span className="ov-text"> {words.slice(words.length - tail).join(' ')}</span>}
        </p>
      </div>
    </div>
  )
}

/* ── part 2: the pipeline board ─────────────────────────────── */
function Board({ map, target, stage, hover, setHover }) {
  const cands = target.candidates
  const byBi = [...cands].sort((a, b) => a.biRank - b.biRank)
  const byCe = [...cands].sort((a, b) => a.ceRank - b.ceRank)
  const p1 = new Set(top3(cands, 'biRank').map((c) => c.chunk))
  const p2 = new Set(top3(cands, 'ceRank').map((c) => c.chunk))
  const shortlist = new Set(cands.map((c) => c.chunk))
  const gold = new Set(target.gold ?? [])
  const y = (r) => (r - 0.5) * ROW
  const hybrid = target.retrieval === 'hybrid'
  const labels = pipelineLabels(target.retrieval, target.reranker)
  const rankNote = (c) => hybrid
    ? `dense ${c.denseRank ? `#${c.denseRank}` : 'not in top 20'} · BM25 ${c.bm25Rank ? `#${c.bm25Rank}` : 'not in top 20'}`
    : undefined

  const row = (c, rank, kept) => (
    <div
      key={c.chunk}
      className={`frow ${gold.has(c.chunk) ? 'gold' : ''} ${hover === c.chunk ? 'hover' : ''} ${stage >= 3 && kept ? 'kept' : ''}`}
      onMouseEnter={() => setHover(c.chunk)}
      onMouseLeave={() => setHover(null)}
      title={rankNote(c)}
    >
      <span className="r">{rank}</span>
      <span className="c">{c.cite}</span>
      <span className="p">#{c.chunk}</span>
    </div>
  )

  const box = (label, set, answer, side) => (
    <div className={`llm-box ${side} ${stage >= 3 ? 'on' : ''}`}>
      <span className="sheet-label">{label}</span>
      {[...set].map((id) => {
        const c = cands.find((x) => x.chunk === id)
        return (
          <span key={id} className={`ctx-chip ${gold.has(id) ? 'gold' : ''} ${hover === id ? 'hover' : ''}`}
            onMouseEnter={() => setHover(id)} onMouseLeave={() => setHover(null)}>
            {c.cite}
          </span>
        )
      })}
      {stage >= 4 && (
        <p className="llm-out">
          {answer ? `${answer.replace(/[*#>_]/g, '').slice(0, 170)}…` : 'No answer generated yet. Ask it in the chat to get both answers.'}
        </p>
      )}
    </div>
  )

  return (
    <div className="board">
      <div className="fcol doc on">
        <span className="sheet-label">Document · {map.chunks.length} chunks</span>
        <div className={`mini-cells ${map.chunks.length > 700 ? 'dense' : ''}`}>
          {map.chunks.map((c, id) => (
            <i
              key={id}
              title={`#${id} · ${c.cite}`}
              className={[
                stage >= 1 && shortlist.has(id) ? 'short' : '',
                stage >= 3 && p2.has(id) ? 'kept' : '',
                gold.has(id) ? 'gold' : '',
                hover === id ? 'hover' : '',
              ].join(' ')}
              onMouseEnter={() => shortlist.has(id) && setHover(id)}
              onMouseLeave={() => setHover(null)}
            />
          ))}
        </div>
      </div>

      <div className={`fcol bi ${stage >= 1 ? 'on' : ''}`}>
        <span className="sheet-label">{hybrid ? 'Fused order · dense + BM25' : 'Bi-encoder order'}</span>
        <div className="flist">{byBi.map((c) => row(c, c.biRank, p1.has(c.chunk)))}</div>
      </div>

      <svg className={`links ${stage >= 2 ? 'on' : ''}`} viewBox={`0 0 100 ${ROW * 10}`} preserveAspectRatio="none" aria-hidden="true">
        {cands.map((c) => (
          <path
            key={c.chunk}
            className={`${gold.has(c.chunk) ? 'gold' : ''} ${hover === c.chunk ? 'hover' : ''} ${c.ceRank < c.biRank ? 'up' : c.ceRank > c.biRank ? 'down' : ''}`}
            d={`M0,${y(c.biRank)} C50,${y(c.biRank)} 50,${y(c.ceRank)} 100,${y(c.ceRank)}`}
            vectorEffect="non-scaling-stroke"
          />
        ))}
      </svg>

      <div className={`fcol ce ${stage >= 2 ? 'on' : ''}`}>
        <span className="sheet-label">Cross-encoder order · {target.reranker}</span>
        <div className="flist">{byCe.map((c) => row(c, c.ceRank, p2.has(c.chunk)))}</div>
      </div>

      <div className={`fcol llm ${stage >= 3 ? 'on' : ''}`}>
        <span className="sheet-label">What the LLM reads</span>
        {box(labels.baseline, p1, target.answers?.baseline, 'bi')}
        {box(labels.reranked, p2, target.answers?.reranked, 'ce')}
      </div>
    </div>
  )
}

/** Chunk map for the active document: live from the server, or from the saved results for the GDPR offline. */
function useDocMap(doc) {
  const { data, server } = useStore()
  const [maps, setMaps] = useState({})
  const id = doc?.id
  const live = server.state === 'live'
  const saved = id === GDPR_ID && data
    ? { chunks: data.chunkIndex, example: data.chunkingExample, chunkWords: data.document.chunkWords, overlapWords: data.document.overlapWords }
    : null
  useEffect(() => {
    if (!id || !live || doc.status !== 'ready' || maps[id]) return
    fetch(`api/documents/${id}/map`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.status))))
      .then((m) => setMaps((all) => ({ ...all, [id]: m })))
      .catch(() => {})
  }, [id, live, doc?.status, maps])
  return maps[id] ?? saved
}

export default function Flow() {
  const { data, study, qi, setQi, turns, flowTarget, setFlowTarget, activeDoc, server } = useStore()
  const [stage, setStage] = useState(4)
  const [hover, setHover] = useState(null)
  const [traces, setTraces] = useState({})      // per document: { query, candidates, timings, retrieval, reranker }
  const [traceText, setTraceText] = useState('')
  const [retrieval, setRetrieval] = useState('dense')
  const [reranker, setReranker] = useState(DEFAULT_RERANKER)
  const [tracing, setTracing] = useState(false)
  const [traceError, setTraceError] = useState(null)
  const [pick, setPick] = useState(null)        // 'trace' | 'chat:<id>' | 'test:<i>'
  const timer = useRef(null)
  useEffect(() => () => clearInterval(timer.current), [])

  const map = useDocMap(activeDoc)
  const docId = activeDoc?.id
  const isCase = !!study   // the active document has a case study with test questions
  const chatTurns = turns.filter((t) => t.docId === docId && t.candidates)
  const trace = traces[docId]

  // What to show: an explicit pick, else the chat turn sent here, else the newest available question.
  const fallback = flowTarget?.kind === 'chat' && chatTurns.some((t) => t.id === flowTarget.id)
    ? `chat:${flowTarget.id}`
    : trace ? 'trace' : chatTurns.length ? `chat:${chatTurns.at(-1).id}` : isCase ? `test:${qi}` : null
  const valid = (p) => p && ((p === 'trace' && trace) || (p.startsWith('chat:') && chatTurns.some((t) => `chat:${t.id}` === p)) || (p.startsWith('test:') && isCase))
  const sel = valid(pick) ? pick : fallback

  let target = null
  if (sel === 'trace') target = { ...trace, gold: [] }
  else if (sel?.startsWith('chat:')) {
    const t = chatTurns.find((c) => `chat:${c.id}` === sel)
    target = {
      query: t.query, candidates: t.candidates, timings: t.timings,
      retrieval: t.retrieval ?? 'dense', reranker: t.reranker ?? DEFAULT_RERANKER,
      gold: t.candidates.filter((c) => c.gold).map((c) => c.chunk),
      answers: t.answers?.reranked && { baseline: t.answers.baseline?.text, reranked: t.answers.reranked.text },
    }
  } else if (sel?.startsWith('test:')) {
    const q = study.queries[+sel.slice(5)]
    target = { query: q.query, candidates: q.candidates, gold: q.gold, answers: q.answers, timings: null,
      retrieval: 'dense', reranker: DEFAULT_RERANKER }   // the case study runs the notebook's pipelines
  }

  const runTrace = async (e) => {
    e.preventDefault()
    const query = traceText.trim()
    if (query.length < 3) return
    setTracing(true)
    setTraceError(null)
    try {
      const r = await fetch('api/retrieve', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query, doc: docId, retrieval, reranker }) })
      const body = await r.json()
      if (!r.ok) throw new Error(body.detail ?? `HTTP ${r.status}`)
      setTraces((all) => ({ ...all, [docId]: { query, candidates: body.candidates, timings: body.timings, retrieval, reranker } }))
      setPick('trace')
      setStage(3)
    } catch (err) {
      setTraceError(err.message)
    } finally {
      setTracing(false)
    }
  }

  const choose = (v) => {
    setPick(v)
    setFlowTarget(null)
    if (v.startsWith('test:')) setQi(+v.slice(5))
  }

  const play = () => {
    clearInterval(timer.current)
    let s = 0
    setStage(0)
    timer.current = setInterval(() => {
      s += 1
      setStage(s)
      if (s >= 4) clearInterval(timer.current)   // the last stage
    }, 1500)
  }

  if (!activeDoc) return null
  const nChunks = map?.chunks.length ?? activeDoc.chunks ?? 0
  const t = target?.timings
  const lat = study?.latency ?? data.latency
  const searchMs = t ? t.embedMs + t.searchMs : lat.retrieveMs
  const rerankMs = t ? t.rerankMs : lat.rerankMs
  const swapped = target
    ? top3(target.candidates, 'ceRank').filter((c) => !top3(target.candidates, 'biRank').some((b) => b.chunk === c.chunk)).length
    : 0
  const k = target?.candidates.length ?? 10
  const hybrid = target?.retrieval === 'hybrid'
  const ce = target?.reranker ?? DEFAULT_RERANKER
  const STAGES = stages(hybrid, ce)
  const captions = [
    `The PDF is cut into ${nChunks} overlapping chunks and each one is embedded once into a 384-number vector. This happens when the document is added, before any question.`,
    hybrid
      ? `The question is embedded with the same bi-encoder and compared with all ${nChunks} vectors, and BM25 keyword search scores the chunks by the words they share with it. Each list keeps its top 20; Reciprocal Rank Fusion merges them and the ${k} best are kept (${fmtDuration(searchMs + (t?.fuseMs ?? 0))}). Hover a chunk to see its place in each list.`
      : `The question is embedded with the same bi-encoder and compared with all ${nChunks} vectors. The ${k} closest chunks are kept (${fmtDuration(searchMs)}).`,
    `The cross-encoder (${ce}) reads the question together with each of the ${k} chunks and scores the pair. The lines show how the order changed (${fmtDuration(rerankMs)}).`,
    `The baseline keeps the ${hybrid ? 'fused' : "bi-encoder's"} top 3; the re-ranked pipeline keeps the cross-encoder's top 3. Here they differ in ${swapped} of 3 chunks.`,
    'Both sets go into the same prompt for the same LLM, which writes Answer 1 and Answer 2.',
  ]

  return (
    <>
      <section className="part wrap">
        <div className="part-head">
          <h2>How a chunk gets to the LLM</h2>
          <p className="doc-line">
            Following <b>{activeDoc.title}</b> · <Link to="/">switch or add a PDF</Link>
          </p>
          <p>
            First, how the document is cut into chunks. Then every step a question goes through, from all{' '}
            {nChunks || 'the'} chunks to the 3 the language model reads.
          </p>
        </div>
        {map ? <Chunking key={docId} map={map} /> : (
          <p className="offline sheet sheet-pad">
            {activeDoc.status !== 'ready'
              ? `This document is still being indexed (${Math.round((activeDoc.progress ?? 0) * 100)}%).`
              : server.state !== 'live'
                ? (STATIC_DEMO ? DEMO_NOTE : 'Start the server to see how this document was chunked.')
                : 'Loading the chunk map…'}
          </p>
        )}
      </section>

      <section className="part wrap" style={{ paddingTop: 48 }}>
        <div className="flow-controls">
          <form className="trace-form" onSubmit={runTrace}>
            <label htmlFor="trace-q" className="hand">Trace a question</label>
            <input
              id="trace-q"
              value={traceText}
              onChange={(e) => setTraceText(e.target.value)}
              placeholder={server.state === 'live' ? 'Type a question; retrieval only, no LLM' : 'Needs the live server'}
              disabled={server.state !== 'live' || activeDoc.status !== 'ready'}
            />
            <button className="btn hot" disabled={tracing || traceText.trim().length < 3 || server.state !== 'live'}>
              {tracing ? 'Tracing…' : 'Trace'}
            </button>
            <PipelinePicker retrieval={retrieval} setRetrieval={setRetrieval} reranker={reranker} setReranker={setReranker} live />
          </form>
          {traceError && <p className="red-note small">{traceError}</p>}
          {(chatTurns.length > 0 || isCase || trace) && (
            <div className="test-select">
              <label htmlFor="flow-q" className="hand">Or follow</label>
              <select id="flow-q" value={sel ?? ''} onChange={(e) => choose(e.target.value)}>
                {trace && <optgroup label="Traced here"><option value="trace">{trace.query}</option></optgroup>}
                {chatTurns.length > 0 && (
                  <optgroup label="Asked in chat">
                    {chatTurns.map((c) => <option key={c.id} value={`chat:${c.id}`}>{c.query}</option>)}
                  </optgroup>
                )}
                {isCase && (
                  <optgroup label="Case-study test questions">
                    {study.queries.map((q, i) => <option key={q.query} value={`test:${i}`}>{q.query}</option>)}
                  </optgroup>
                )}
              </select>
            </div>
          )}
        </div>

        {target && map ? (
          <div className="sheet sheet-pad flow-sheet">
            <div className="stages" role="group" aria-label="Pipeline stage">
              {STAGES.map(([name, sub], i) => (
                <button key={name} aria-pressed={stage === i} className={stage >= i ? 'reached' : ''}
                  onClick={() => { clearInterval(timer.current); setStage(i) }}>
                  <b>{i + 1}</b>
                  <span>{name}</span>
                  <small>{sub}</small>
                </button>
              ))}
              <button className="btn hot play" onClick={play}>▶ Play</button>
            </div>
            <p className="trace-q-line">Q: {target.query}</p>
            <p className="stage-caption">{captions[stage]}</p>
            <Board map={map} target={target} stage={stage} hover={hover} setHover={setHover} />
            <p className="board-legend">
              {isCase && <span><i className="lg gold" />holds the answer ({study?.generated ? 'generated' : 'labelled'} test questions)</span>}
              <span><i className="lg kept" />kept for the LLM</span>
              <span><i className="lg up" />climbed</span>
              <span><i className="lg down" />fell</span>
              <span>Hover any chunk to trace it across columns.</span>
            </p>
          </div>
        ) : (
          <div className="sheet sheet-pad flow-empty">
            <p className="hand">Type a question above to watch it travel through the pipeline.</p>
            <p>Retrieval only takes about a second and uses no LLM quota. Questions you ask in the chat show up here too.</p>
          </div>
        )}
      </section>
    </>
  )
}
