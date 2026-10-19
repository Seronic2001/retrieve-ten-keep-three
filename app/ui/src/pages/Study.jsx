import { useState } from 'react'
import { Link, NavLink, Outlet } from 'react-router-dom'
import { GDPR_ID, useStore } from '../store.jsx'
import { DEMO_NOTE, STATIC_DEMO, fmtDuration } from '../lib.js'
import Hero from '../components/Hero.jsx'

const SUB = [
  ['', 'Overview'],
  ['models', 'Two readers'],
  ['replay', 'Re-rank replay'],
  ['results', 'Results'],
  ['cost', 'Cost'],
]

const STAGES = {
  writing: 'Writing test questions',
  retrieving: 'Running both pipelines',
  answering: 'Generating both answers',
  timing: 'Timing a full cross-encoder scan',
}

/** Starts a generated case study for an uploaded document and shows its progress. */
function StudyBuilder() {
  const { activeDoc, studyState, buildStudy, server } = useStore()
  const [backend, setBackend] = useState(server.local ? 'local' : 'gemini')
  const [err, setErr] = useState(null)
  const live = server.state === 'live'
  const building = studyState.state === 'building'
  const pct = building ? Math.round(((Object.keys(STAGES).indexOf(studyState.stage) + (studyState.total ? studyState.done / studyState.total : 0)) / 4) * 100) : 0
  const start = () => { setErr(null); buildStudy(backend).catch((e) => setErr(e.message)) }

  return (
    <section className="part wrap">
      <div className="sheet sheet-pad study-build">
        <span className="sheet-label">No case study yet</span>
        <h2>Make <b>{activeDoc.title}</b> a case study</h2>
        <p>
          The GDPR study uses 13 hand-labelled questions. For this document, a language model writes about 12 test
          questions from passages spread across it, each in one of the same styles (keyword trap, exception, negation,
          buried fact, control), and quotes the words that answer it. Every chunk holding those words counts as the
          answer, so both pipelines can be scored the same way. Both answers are generated where the two pipelines
          give the LLM different context.
        </p>
        {!live ? (
          <p className="offline">{STATIC_DEMO ? DEMO_NOTE : 'Start the server to build a case study.'}</p>
        ) : activeDoc.status !== 'ready' ? (
          <p className="offline">This document is still being indexed.</p>
        ) : building ? (
          <div className="doc-progress">
            <div className="meter"><i style={{ width: `${Math.max(3, pct)}%` }} /></div>
            <span className="mono">
              {STAGES[studyState.stage] ?? 'Starting'}
              {studyState.total > 1 && ` · ${studyState.done + 1} of ${studyState.total}`}
            </span>
          </div>
        ) : (
          <div className="study-build-go">
            <div className="tabs" role="group" aria-label="Language model">
              <button aria-pressed={backend === 'gemini'} onClick={() => setBackend('gemini')}>Gemini · one call for the questions</button>
              <button aria-pressed={backend === 'local'} disabled={!server.local} onClick={() => setBackend('local')}
                title={server.local ? server.local : 'Start the local model in WSL (bash start_local_llm.sh) or set LOCAL_LLM_URL'}>
                Local model · one call per question
              </button>
            </div>
            <button className="btn hot" onClick={start}>Build case study</button>
            <span className="side-note">Takes one to three minutes.</span>
          </div>
        )}
        {studyState.state === 'error' && !building && <p className="red-note small">{studyState.error}</p>}
        {err && <p className="red-note small">{err}</p>}
      </div>
    </section>
  )
}

/** The evaluation of one document: test questions with known answer chunks, scored on both pipelines. */
export default function Study() {
  const { activeDoc, study, studyState, server } = useStore()
  if (!activeDoc) return null
  const isCase = activeDoc.id === GDPR_ID
  const label = isCase
    ? 'Case study · GDPR (EU) 2016/679 · 13 labelled test questions'
    : `Case study · ${activeDoc.title}${study ? ` · ${study.queries.length} generated test questions` : ''}`
  return (
    <>
      <div className="study-bar">
        <div className="wrap">
          <span className="study-label">{label}</span>
          {study && (
            <nav className="subnav" aria-label="Case study sections">
              {SUB.map(([to, name]) => (
                <NavLink key={to} to={to === '' ? '/study' : `/study/${to}`} end={to === ''}>{name}</NavLink>
              ))}
            </nav>
          )}
        </div>
      </div>
      {study ? <Outlet key={activeDoc.id} /> : studyState.state === 'unknown' && server.state === 'live' && activeDoc.status === 'ready' ? <p className="loading">Loading…</p> : <StudyBuilder />}
    </>
  )
}

export function StudyOverview() {
  const { study, qi, activeDoc, studyState, buildStudy, server } = useStore()
  const [err, setErr] = useState(null)
  const isCase = activeDoc.id === GDPR_ID
  const count = (s) => study.queries.filter((q) => q.status === s).length
  const gen = study.generated
  const cards = [
    ['/study/replay', 'Re-rank replay', 'Watch the ten candidates swap places for each test question, then read both answers.', 'the core result'],
    ['/study/results', 'Scoreboard', `Every test question, rescued or broken${study.sweep ? `, plus a ${study.sweep.questions}-question stress test` : ''}.`,
      `${count('restored')} rescued · ${count('degraded')} broken`],
    ['/study/models', 'Two readers', 'How a bi-encoder and a cross-encoder read the same question and passage.', 'hover to compare'],
    ['/study/cost', 'Cost', 'What re-ranking ten candidates costs, and why nobody re-ranks the whole corpus.', `+${fmtDuration(study.latency.rerankMs)}`],
  ]
  const rebuild = () => { setErr(null); buildStudy(gen.backend).catch((e) => setErr(e.message)) }
  return (
    <>
      <Hero data={study} query={study.queries[qi]} />
      <section className="part wrap" style={{ paddingTop: 24 }}>
        <div className="cards">
          {cards.map(([to, title, text, note]) => (
            <Link key={to} to={to} className="sheet sheet-pad card">
              <span className="card-note">{note}</span>
              <h3>{title}</h3>
              <p>{text}</p>
              <span className="card-go">Open →</span>
            </Link>
          ))}
        </div>
        {gen && (
          <p className="side-note study-gen">
            Questions written by {gen.questionsBy} on {new Date(gen.at).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })};
            answer chunks found by the quoted words, not checked by hand.{gen.note && <> {gen.note}</>}{' '}
            {studyState.state === 'building'
              ? 'Rebuilding…'
              : server.state === 'live' && <button className="linkish" onClick={rebuild}>Write new questions</button>}
            {err && <span className="red-note"> {err}</span>}
          </p>
        )}
        <p className="study-ask">
          <Link to="/chat" className="btn hot">Ask {isCase ? 'the GDPR' : 'this document'} your own question →</Link>
        </p>
      </section>
    </>
  )
}
