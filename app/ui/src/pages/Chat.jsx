import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import Markdown from 'react-markdown'
import { GDPR_ID, top3, useStore } from '../store.jsx'
import { STATIC_DEMO, fmtDuration } from '../lib.js'
import AnswerChecks from '../components/AnswerChecks.jsx'

const GDPR_OWN = [
  'Can my employer read my work emails?',
  'Do I need consent to send customers a newsletter?',
  'What must a company do after leaking customer data?',
]
// Starters that make sense for most documents; retrieval works best on specific questions.
const GENERIC = [
  'What are the main rules or requirements in this document?',
  'Are there any deadlines or time limits?',
  'What exceptions or special cases are mentioned?',
  'What happens if the rules are not followed?',
]

function Trace({ turn }) {
  const { library } = useStore()
  const nChunks = library.find((d) => d.id === turn.docId)?.chunks
  const k = turn.candidates?.length ?? 10
  const t = turn.timings
  const nAnswers = Object.keys(turn.answers ?? {}).length
  const steps = [
    [nChunks ? `Searched ${nChunks} chunks` : 'Searched the document', turn.candidates ? (t ? fmtDuration(t.embedMs + t.searchMs) : 'saved') : null, !turn.candidates],
    [`Re-ranked the top ${k}`, turn.candidates ? (t ? fmtDuration(t.rerankMs) : 'saved') : null, !turn.candidates],
    [turn.status === 'answering' ? 'Writing' : nAnswers === 1 ? 'Wrote the answer' : nAnswers ? `Wrote ${nAnswers} answers` : 'Answer',
      turn.status === 'answering'
        ? (turn.answers?.baseline?.streaming ? 'streaming baseline' : turn.answers?.reranked?.streaming ? 'streaming re-ranked' : 'waiting for model')
        : turn.answers?.reranked?.saved
          ? 'from notebook run'
          : nAnswers
            ? fmtDuration(Object.values(turn.answers).reduce((s, a) => s + (a.ms ?? 0), 0))
            : 'skipped',
      turn.status === 'answering'],
  ]
  return (
    <ol className="trace">
      {steps.map(([label, value, busy]) => (
        <li key={label} className={busy ? 'busy' : value === 'skipped' ? 'skip' : 'done'}>
          <span className="tick" aria-hidden="true">{busy ? '' : value === 'skipped' ? '–' : '✓'}</span>
          {label}
          {value && <span className="mono">{value}</span>}
        </li>
      ))}
    </ol>
  )
}

function Sources({ chunks, onOpen }) {
  return (
    <div className="sources">
      {chunks.map((c, i) => (
        <button key={c.chunk} className={`source ${c.gold ? 'gold' : ''}`} onClick={() => onOpen(c)}>
          <span className="n">{i + 1}</span>
          {c.cite}{!c.cite.startsWith('p') && <small>p. {c.page}</small>}
        </button>
      ))}
    </div>
  )
}

function AssistantTurn({ turn, onOpen }) {
  const { setFlowTarget, setQi, answerMore } = useStore()
  const navigate = useNavigate()
  // Side by side once a baseline answer exists or is on its way, unless the reader picked a tab.
  const [chosen, setView] = useState(null)
  const view = chosen ?? (turn.answers?.baseline || turn.pending?.includes('baseline') ? 'compare' : 'reranked')

  if (turn.status === 'error') return <p className="red-note">{turn.error}</p>
  if (!turn.candidates) return <Trace turn={turn} />

  const base = top3(turn.candidates, 'biRank')
  const rer = top3(turn.candidates, 'ceRank')
  const swapped = rer.filter((c) => !base.some((b) => b.chunk === c.chunk)).length
  const panes = { baseline: ['Baseline · Pipeline 1', base], reranked: ['Re-ranked · Pipeline 2', rer] }
  const shown = view === 'compare' ? ['baseline', 'reranked'] : [view]

  return (
    <div className="bot">
      <Trace turn={turn} />
      <p className="swap-note">
        {swapped === 0
          ? 'Re-ranking kept the same three chunks.'
          : `Re-ranking swapped ${swapped} of the 3 chunks the LLM reads.`}
        {turn.source === 'saved' && <span className="tag">saved results · server offline</span>}
      </p>

      <div className="tabs" role="group" aria-label="Answer view">
        <button aria-pressed={view === 'reranked'} onClick={() => setView('reranked')}>Re-ranked</button>
        <button aria-pressed={view === 'baseline'} onClick={() => setView('baseline')}>Baseline</button>
        <button aria-pressed={view === 'compare'} onClick={() => setView('compare')}>Side by side</button>
      </div>

      <div className={`answers ${view === 'compare' ? 'two' : ''}`}>
        {shown.map((k) => {
          const a = turn.answers?.[k]
          return (
          <div key={k} className={`answer ${k}`}>
            <span className="sheet-label">
              {panes[k][0]}
              {a && !a.streaming && a.ms != null && (
                <span className="ans-time"> · first token {fmtDuration(a.firstTokenMs)} · done in {fmtDuration(a.ms)}</span>
              )}
            </span>
            {a ? (
              <>
                <div className={`ans-body ${a.streaming ? 'live' : ''}`}>
                  {a.text ? <Markdown>{a.text}</Markdown> : <p className="writing">reading the 3 chunks…</p>}
                </div>
                {!a.streaming && <AnswerChecks text={a.text} chunks={panes[k][1]} />}
              </>
            ) : turn.status === 'answering' && turn.pending?.includes(k) ? (
              <p className="writing">
                {k === 'baseline' ? 'queued, streams after the re-ranked answer…' : 'waiting for the model…'}
              </p>
            ) : turn.backend && turn.source === 'live' && turn.answers ? (
              <div className="not-generated">
                <p>
                  Not generated yet. The {k === 'baseline' ? 'baseline' : 're-ranked'} answer uses the
                  {k === 'baseline' ? ' bi-encoder' : ' cross-encoder'} top 3 below.
                </p>
                <button className="btn" disabled={turn.status === 'answering'}
                  onClick={() => { setView('compare'); answerMore(turn, k) }}>
                  Generate {k === 'baseline' ? 'baseline' : 're-ranked'} answer
                </button>
              </div>
            ) : (
              <ul className="previews">
                {panes[k][1].map((c) => <li key={c.chunk}><b>{c.cite}</b> {c.text.slice(0, 180)}…</li>)}
              </ul>
            )}
            <Sources chunks={panes[k][1]} onOpen={onOpen} />
          </div>
          )
        })}
      </div>
      {turn.note && <p className="red-note small">{turn.note}</p>}

      <div className="turn-actions">
        <button className="btn" onClick={() => { setFlowTarget({ kind: 'chat', id: turn.id }); navigate('/flow') }}>
          Follow these chunks →
        </button>
        {turn.testIndex != null && (
          <button className="btn" onClick={() => { setQi(turn.testIndex); navigate('/study/replay') }}>Open in replay →</button>
        )}
        {turn.answers?.reranked && (
          <span className="model-tag mono">{turn.answers.reranked.model}{turn.answers.reranked.saved ? ' · saved' : ''}</span>
        )}
      </div>
    </div>
  )
}

function ChunkDrawer({ chunk, onClose }) {
  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <div className="drawer-scrim" onClick={onClose}>
      <aside className="sheet drawer" onClick={(e) => e.stopPropagation()} aria-label="Chunk text">
        <div className="chunk-head">
          <h3>{chunk.cite}{!chunk.cite.startsWith('p') && ` · p. ${chunk.page}`}</h3>
          <button className="btn" onClick={onClose}>Close</button>
        </div>
        <div className="chunk-scores">
          <span>bi <b>#{chunk.biRank}</b> ({chunk.biScore.toFixed(3)})</span>
          <span>cross <b>#{chunk.ceRank}</b> ({chunk.ceScore.toFixed(2)})</span>
          <span>chunk <b>#{chunk.chunk}</b></span>
          {chunk.gold && <span className="gold-tag">the answer</span>}
        </div>
        <p className="legal">{chunk.text}</p>
      </aside>
    </div>
  )
}

export default function Chat() {
  const { study, turns: allTurns, ask, server, clearChat, activeDoc } = useStore()
  const [text, setText] = useState('')
  const [mode, setMode] = useState('answer')   // 'answer' (re-ranked only) | 'compare' | 'chunks'
  // The local model is the default; when llama-server is not running, Gemini is used instead.
  const [backend, setBackend] = useState('local')
  const llm = backend === 'local' && server.local ? 'local' : 'gemini'
  const [open, setOpen] = useState(null)
  const isCase = activeDoc?.id === GDPR_ID
  const turns = allTurns.filter((t) => t.docId === activeDoc?.id)   // each document keeps its own thread
  const busy = allTurns.some((t) => t.status === 'retrieving' || t.status === 'answering')
  const canAsk = activeDoc?.status === 'ready' && (server.state === 'live' || isCase)

  // Follow new text while the reader is at the bottom; stop following once they scroll up to read.
  const stick = useRef(true)
  const count = useRef(0)
  useEffect(() => {
    const onScroll = () => {
      stick.current = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 160
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])
  useEffect(() => {
    const isNew = turns.length > count.current
    count.current = turns.length
    if (!turns.length) return
    if (isNew) stick.current = true
    // Scroll to the page bottom, where the sticky composer rests below the thread instead of covering its
    // last lines. Instant: a smooth scroll would fire mid-way scroll events and unstick.
    if (stick.current) window.scrollTo(0, document.documentElement.scrollHeight)
  }, [turns])

  const send = (q) => {
    const query = (q ?? text).trim()
    if (query.length < 3 || busy || !canAsk) return
    setText('')
    ask(query, { withLlm: mode !== 'chunks', compare: mode === 'compare', backend: llm })
  }
  const suggestions = study
    ? [...study.queries.filter((q) => q.answers).map((q) => q.query).slice(0, 4), ...(isCase && server.state === 'live' ? GDPR_OWN : [])]
    : GENERIC

  if (!activeDoc) return null
  return (
    <section className="part wrap chat-page">
      <div className="part-head">
        <h2>Ask the document</h2>
        <p className="doc-line">
          Reading <b>{activeDoc.title}</b>
          {activeDoc.pages != null && <> · {activeDoc.pages} pages</>}
          {activeDoc.chunks != null && <> · {activeDoc.chunks} chunks</>}
          {' · '}<Link to="/">switch or add a PDF</Link>
        </p>
        <p>
          Every question is retrieved both ways. You get the answer written from the re-ranked chunks, and can add
          the baseline answer to compare. Quotes in each answer are checked against the chunks it read.
        </p>
      </div>

      {server.state === 'offline' && (STATIC_DEMO ? (
        <p className="offline sheet sheet-pad">
          <b>Hosted demo.</b> Pick one of the 13 GDPR test questions to see both pipelines&apos; retrievals and saved
          answers. Asking new questions needs the full app with its server.
        </p>
      ) : (
        <p className="offline sheet sheet-pad">
          <b>Live server offline.</b> {isCase ? 'The 13 GDPR test questions still work from saved results.' : ''} For
          new questions, run <code>.venv/Scripts/python -m uvicorn server:app --port 8000</code> in the project folder
          and reload.
        </p>
      ))}
      {activeDoc.status !== 'ready' && (
        <p className="offline sheet sheet-pad">
          <b>This document is still being indexed</b> ({Math.round((activeDoc.progress ?? 0) * 100)}%). Questions open
          as soon as it is ready.
        </p>
      )}

      <div className="sheet notepad">
        {turns.length === 0 ? (
          <div className="chat-empty">
            <p className="hand">
              {isCase ? 'Ask anything about the GDPR, or start with one of these:' : 'Ask a specific question, or start with one of these:'}
            </p>
            <div className="suggest">
              {suggestions.map((s) => <button key={s} className="slip" disabled={!canAsk} onClick={() => send(s)}>{s}</button>)}
            </div>
            <p className="ask-tips">
              Works best for questions about what the document says. Each answer is written from 3 passages, so it
              cannot count things across the whole document, and it only knows what the document itself contains.
            </p>
          </div>
        ) : (
          turns.map((t) => (
            <div key={t.id} className="turn">
              <p className="me">{t.query}</p>
              <AssistantTurn turn={t} onOpen={setOpen} />
            </div>
          ))
        )}
      </div>

      <form className="composer" onSubmit={(e) => { e.preventDefault(); send() }}>
        <label htmlFor="ask" className="sr-only">Your question</label>
        <textarea
          id="ask"
          rows={2}
          value={text}
          placeholder={isCase ? 'e.g. Can a company keep my data after I close my account?' : `Ask about ${activeDoc.title}…`}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
        />
        <div className="composer-bar">
          <div className="tabs" role="group" aria-label="Answer mode">
            <button type="button" aria-pressed={mode === 'answer'} onClick={() => setMode('answer')}>Answer</button>
            <button type="button" aria-pressed={mode === 'compare'} onClick={() => setMode('compare')}>Compare both</button>
            <button type="button" aria-pressed={mode === 'chunks'} onClick={() => setMode('chunks')}>Chunks only</button>
          </div>
          {mode !== 'chunks' && (
            <div className="tabs" role="group" aria-label="Language model">
              <button type="button" aria-pressed={llm === 'gemini'} onClick={() => setBackend('gemini')}>Gemini</button>
              <button
                type="button"
                aria-pressed={llm === 'local'}
                disabled={!server.local}
                title={server.local ? server.local : 'Start it in WSL (bash start_local_llm.sh) or set LOCAL_LLM_URL'}
                onClick={() => setBackend('local')}
              >
                {server.localHosted ? 'Open model · hosted' : 'Qwen 4B · local'}
              </button>
            </div>
          )}
          <span className="composer-hint">
            {mode === 'chunks'
              ? 'Retrieval only: no LLM, no quota, about a second'
              : `${mode === 'compare' ? 'Re-ranked and baseline answers' : 'Re-ranked answer; add the baseline later to compare'}${
                llm === 'local'
                  ? ` · ${server.local}${server.localHosted ? ', hosted' : ' on your GPU, no quota'}`
                  : `${backend === 'local'
                    ? (server.localHosted ? ' · open model unreachable, using Gemini' : ' · local model offline (bash start_local_llm.sh), using Gemini')
                    : ''} · ${
                    mode === 'compare' ? 'two Gemini calls' : 'one Gemini call'} (free tier: 20 a day per model)`}`}
          </span>
          {turns.length > 0 && <button type="button" className="btn" onClick={() => clearChat(activeDoc.id)}>Clear</button>}
          <button type="submit" className="btn hot" disabled={busy || !canAsk || text.trim().length < 3}>
            {busy ? 'Working…' : 'Ask ↵'}
          </button>
        </div>
      </form>

      {open && <ChunkDrawer chunk={open} onClose={() => setOpen(null)} />}
    </section>
  )
}
