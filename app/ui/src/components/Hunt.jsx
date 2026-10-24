import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { GDPR_ID, useStore } from '../store.jsx'
import { CATEGORY, DEMO_NOTE, STATIC_DEMO, VERDICT, byBi, byCe, markSnippets, rankText } from '../lib.js'
import { RankTrack, Waffle } from './Scoreboard.jsx'

const STAGES = { writing: 'Writing questions', retrieving: 'Running both pipelines' }
const pct = (n, d) => (d ? `${((100 * n) / d).toFixed(1)}%` : '–')
const sameText = (a, b) => a.trim().toLowerCase() === b.trim().toLowerCase()

/** A failure hunt for the active document: loads it, polls while it runs, and starts or redoes it. */
function useHunt(doc, live) {
  const [entry, setEntry] = useState(null)
  const id = doc?.id
  const load = useCallback(() => {
    if (!id) return Promise.resolve()
    return fetch(`api/documents/${id}/hunt`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.status))))
      .then((h) => setEntry({ id, ...h }))
      .catch((e) => setEntry((old) => (old?.id === id && old.hunt ? old : { id, state: 'error', error: `Could not load the failure hunt (${e.message}).` })))
  }, [id])
  const mine = entry?.id === id ? entry : null
  const building = mine?.state === 'building'
  useEffect(() => {
    if (!live || doc?.status !== 'ready') return
    load()
    if (!building) return
    const timer = setInterval(load, 2000)
    return () => clearInterval(timer)
  }, [live, doc?.status, building, load])
  const start = async (backend) => {
    const r = await fetch(`api/documents/${id}/hunt`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ backend }),
    })
    const body = await r.json().catch(() => ({}))
    if (!r.ok) throw new Error(body.detail ?? `HTTP ${r.status}`)
    setEntry((e) => ({ ...(e?.id === id ? e : {}), id, ...body }))
  }
  const markAdded = (i) => setEntry((e) => {
    const questions = e.hunt.questions.map((q, j) => (j === i ? { ...q, added: true } : q))
    return { ...e, hunt: { ...e.hunt, questions } }
  })
  return { entry: mine, start, markAdded }
}

function Tally({ summary }) {
  const rows = [['restored', 'Rescued'], ['degraded', 'Broken'], ['bothHit', 'Both find it'], ['bothMiss', 'Both miss it']]
  const cols = [['natural', 'Ordinary questions'], ['trap', 'Wording traps'], ['all', 'All']]
  return (
    <div className="sheet ledger hunt-tally">
      <table>
        <thead>
          <tr><th />{cols.map(([k, name]) => <th key={k}>{name}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map(([k, name]) => (
            <tr key={k} className={k === 'degraded' ? 'hl-row' : ''}>
              <td className="cat">{name}</td>
              {cols.map(([c]) => (
                <td key={c} className="mono">
                  <b>{summary[c][k]}</b> <span className="of">{pct(summary[c][k], summary[c].questions)}</span>
                </td>
              ))}
            </tr>
          ))}
          <tr>
            <td className="cat">Questions</td>
            {cols.map(([c]) => <td key={c} className="mono"><b>{summary[c].questions}</b></td>)}
          </tr>
        </tbody>
      </table>
    </div>
  )
}

function Finding({ q, i, canAdd, onAdd, onOpen, inStudy }) {
  const [busy, setBusy] = useState(false)
  const [note, setNote] = useState(null)
  const gold = new Set(q.gold)
  const answer = q.candidates.filter((c) => gold.has(c.chunk)).sort((a, b) => a.biRank - b.biRank)[0]
  // Broken: the chunks that took the answer's place in the re-ranked top 3. Rescued: the ones it displaced.
  const took = (q.status === 'degraded' ? byCe(q) : byBi(q)).slice(0, 3).filter((c) => !gold.has(c.chunk))
  const rival = took[0]
  const add = async () => {
    setBusy(true)
    setNote(null)
    try { setNote(await onAdd(i)) } catch (e) { setNote(e.message) } finally { setBusy(false) }
  }
  return (
    <article className={`sheet sheet-pad finding ${q.status}`}>
      <div className="finding-top">
        <span className="cat">{CATEGORY[q.category] ?? q.category}</span>
        <span className={`verdict-word ${q.status}`}>{VERDICT[q.status]}</span>
      </div>
      <p className="finding-q">{q.query}</p>
      <div className="finding-rank">
        <RankTrack from={q.baselineRank} to={q.rerankedRank} status={q.status} />
        <span className="mono">answer chunk {rankText(q.baselineRank)} → {rankText(q.rerankedRank)}</span>
      </div>
      <p className="finding-why">
        Answer in <b className="gold">{answer?.cite ?? `chunk ${q.source}`}</b>
        {took.length > 0 && <>; {q.status === 'degraded' ? 'pushed out by' : 'the baseline read'} <b>{took.map((c) => c.cite).join(', ')}</b> instead</>}
      </p>
      <details>
        <summary>Read the answer chunk{rival ? ' and the one ranked above it' : ''}</summary>
        {answer && (
          <div className="finding-text">
            <span className="sheet-label">{answer.cite} · the answer · bi #{answer.biRank} · cross #{answer.ceRank}</span>
            <p className="legal">
              {markSnippets(answer.text, q.goldSnippets).map((p, k) => (p.hit ? <mark key={k}>{p.text}</mark> : <span key={k}>{p.text}</span>))}
            </p>
          </div>
        )}
        {rival && (
          <div className="finding-text">
            <span className="sheet-label">{rival.cite} · bi #{rival.biRank} · cross #{rival.ceRank}</span>
            <p className="legal">{rival.text}</p>
          </div>
        )}
      </details>
      <div className="finding-actions">
        {inStudy ? (
          <button className="btn" onClick={() => onOpen(q.query)}>In the case study · open replay →</button>
        ) : (
          <button className="btn" disabled={!canAdd || busy} onClick={add}
            title={canAdd ? 'Adds it to the test questions and generates both answers' : 'Build this document\'s case study first'}>
            {busy ? 'Adding and answering…' : 'Add to case study'}
          </button>
        )}
        {note && <span className="side-note">{note}</span>}
      </div>
    </article>
  )
}

/** Failure hunt: a few hundred generated questions, scored on both pipelines, to find where re-ranking breaks. */
export default function Hunt() {
  const { activeDoc, server, study, refreshStudy, setQi } = useStore()
  const live = server.state === 'live'
  const { entry, start, markAdded } = useHunt(activeDoc, live)
  const [backend, setBackend] = useState(server.local ? 'local' : 'gemini')
  const [tab, setTab] = useState('degraded')
  const [limit, setLimit] = useState(12)
  const [err, setErr] = useState(null)
  const navigate = useNavigate()
  const hunt = entry?.hunt
  const building = entry?.state === 'building'
  const isGdpr = activeDoc?.id === GDPR_ID

  const run = () => { setErr(null); start(backend).catch((e) => setErr(e.message)) }
  const open = (query) => {
    const i = study.queries.findIndex((q) => sameText(q.query, query))
    if (i >= 0) { setQi(i); navigate('/study/replay') }
  }
  const add = async (i) => {
    const r = await fetch(`api/documents/${activeDoc.id}/study/add`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ index: i }),
    })
    const body = await r.json().catch(() => ({}))
    if (!r.ok) throw new Error(body.detail ?? `HTTP ${r.status}`)
    markAdded(i)
    await refreshStudy()
    return body.note
  }

  const findings = hunt ? hunt.questions.map((q, i) => [q, i]).filter(([q]) => q.status === tab) : []
  const all = hunt?.summary.all
  const progress = building && entry.total ? Math.round((100 * entry.done) / entry.total) : 0

  return (
    <section id="hunt" className="part wrap">
      <div className="part-head">
        <h2>Failure hunt</h2>
        <p>
          Re-ranking rarely makes things worse: the answer must already be in the bi-encoder&apos;s top 3 and the
          cross-encoder must push it out. To find those cases, a language model writes two questions for each of{' '}
          {hunt?.generated.passages ?? 'about 120'} passages: an ordinary one, and a <b>wording trap</b> that borrows a
          phrase from the most similar other passage, the pattern behind the GDPR&apos;s broken case. Traps are counted
          apart, so the ordinary questions still give a fair failure rate.
        </p>
      </div>

      {!live && <p className="offline sheet sheet-pad">{STATIC_DEMO ? DEMO_NOTE : 'Start the server to run a failure hunt.'}</p>}
      {live && activeDoc.status !== 'ready' && (
        <p className="offline sheet sheet-pad">
          This document is being indexed ({Math.round((activeDoc.progress ?? 0) * 100)}%). Its failure hunt loads when that finishes.
        </p>
      )}
      {live && activeDoc.status === 'ready' && !entry && <p className="side-note">Loading the failure hunt…</p>}

      {live && building && (
        <div className="sheet sheet-pad hunt-run">
          <div className="doc-progress">
            <div className="meter"><i style={{ width: `${Math.max(3, entry.stage === 'retrieving' ? 50 + progress / 2 : progress / 2)}%` }} /></div>
            <span className="mono">{STAGES[entry.stage] ?? 'Starting'}{entry.total > 1 && ` · ${entry.done + 1} of ${entry.total}`}</span>
          </div>
          <p className="side-note">
            {entry.backend === 'local'
              ? 'The local model writes one passage\'s questions per call; this takes a few minutes.'
              : 'Gemini calls are spaced 13 s apart to stay under the free-tier rate limit; this takes a few minutes.'}
          </p>
        </div>
      )}

      {live && !building && !hunt && entry && (
        <div className="sheet sheet-pad hunt-run">
          <div className="study-build-go">
            <div className="tabs" role="group" aria-label="Language model">
              <button aria-pressed={backend === 'gemini'} onClick={() => setBackend('gemini')}>Gemini · 240 questions, 12 calls</button>
              <button aria-pressed={backend === 'local'} disabled={!server.local} onClick={() => setBackend('local')}
                title={server.local ? server.local : 'Start the local model in WSL (bash start_local_llm.sh) or set LOCAL_LLM_URL'}>
                Local model · 40 questions
              </button>
            </div>
            <button className="btn hot" onClick={run}>Run failure hunt</button>
          </div>
        </div>
      )}
      {entry?.state === 'error' && <p className="red-note small">{entry.error}</p>}
      {err && <p className="red-note small">{err}</p>}

      {hunt && (
        <>
          <div className="hunt-head">
            <div className="hunt-big">
              <b>{all.degraded}</b>
              <span>broken of {all.questions}<br />({pct(hunt.summary.natural.degraded, hunt.summary.natural.questions)} of ordinary questions)</span>
            </div>
            <div className="hunt-big">
              <b className="r">{all.restored}</b>
              <span>rescued<br />({pct(hunt.summary.natural.restored, hunt.summary.natural.questions)} of ordinary questions)</span>
            </div>
            <Waffle restored={all.restored} degraded={all.degraded} total={all.questions} />
          </div>
          <Tally summary={hunt.summary} />

          <div className="tabs hunt-tabs" role="group" aria-label="Show">
            <button aria-pressed={tab === 'degraded'} onClick={() => { setTab('degraded'); setLimit(12) }}>Broken ({all.degraded})</button>
            <button aria-pressed={tab === 'restored'} onClick={() => { setTab('restored'); setLimit(12) }}>Rescued ({all.restored})</button>
          </div>
          {findings.length === 0 ? (
            <p className="sheet sheet-pad side-note">
              {tab === 'degraded' ? 'No broken cases in this hunt. Run it again for new questions, or treat it as a result: re-ranking did not hurt here.' : 'No rescued cases in this hunt.'}
            </p>
          ) : (
            <div className="findings">
              {findings.slice(0, limit).map(([q, i]) => (
                <Finding key={i} q={q} i={i} onAdd={add} onOpen={open}
                  canAdd={isGdpr || !!study} inStudy={!!study?.queries.some((s) => sameText(s.query, q.query))} />
              ))}
            </div>
          )}
          {findings.length > limit && <button className="btn" onClick={() => setLimit(findings.length)}>Show all {findings.length}</button>}

          <p className="side-note hunt-foot">
            Questions written by {hunt.generated.questionsBy} on{' '}
            {new Date(hunt.generated.at).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })}.
            Every question is saved in <span className="mono">{isGdpr ? 'notebook/results/gdpr_hunt.csv' : `app/uploads/${activeDoc.id}.hunt.csv`}</span>.
            {hunt.generated.note && <> {hunt.generated.note}</>}
            {hunt.generated.goldCorrected && <> {hunt.generated.goldCorrected}</>}{' '}
            {!building && live && <button className="linkish" onClick={() => start(hunt.generated.backend).catch((e) => setErr(e.message))}>Run again with new questions</button>}
          </p>
        </>
      )}
    </section>
  )
}
