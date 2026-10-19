import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { STATIC_DEMO } from './lib.js'

const Store = createContext(null)
export const useStore = () => useContext(Store)

export const GDPR_ID = 'gdpr'
const ACTIVE_KEY = 'rag.activeDoc'

const sameText = (a, b) => a.trim().toLowerCase().replace(/\s+/g, ' ') === b.trim().toLowerCase().replace(/\s+/g, ' ')

async function post(path, body) {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}))
    const err = new Error(detail.detail ?? `HTTP ${r.status}`)
    err.status = r.status
    throw err
  }
  return r.json()
}

export const top3 = (cands, key) => [...cands].sort((a, b) => a[key] - b[key]).slice(0, 3)

const readActive = () => { try { return localStorage.getItem(ACTIVE_KEY) || GDPR_ID } catch { return GDPR_ID } }
const writeActive = (id) => { try { localStorage.setItem(ACTIVE_KEY, id) } catch { /* storage blocked: keep in memory only */ } }

/**
 * App-wide state: the exported GDPR case-study results, the document library, the active document,
 * the live server status, and the chat thread (kept here so it survives page changes).
 */
export function StoreProvider({ children }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [qis, setQis] = useState({})               // selected test question, per document
  const [studies, setStudies] = useState({})       // generated case studies of uploads, by document id
  const [server, setServer] = useState({ state: 'checking' })
  const [docs, setDocs] = useState(null)           // from the server; null until the first answer
  const [activeId, setActiveIdRaw] = useState(readActive)
  const [uploading, setUploading] = useState(null)  // { name, pct } while bytes are being sent
  const [turns, setTurns] = useState([])
  const [flowTarget, setFlowTarget] = useState(null) // { kind: 'chat', id } or null for the test question
  const nextId = useRef(1)
  const activateWhenReady = useRef(null)

  const setActiveId = useCallback((id) => { setActiveIdRaw(id); writeActive(id); setFlowTarget(null) }, [])

  useEffect(() => {
    fetch('data.json')
      .then((r) => r.json())
      .then((d) => {
        setData(d)
        const start = d.queries.findIndex((q) => q.query.startsWith('What is the maximum fine'))
        setQis((all) => ({ [GDPR_ID]: start >= 0 ? start : 0, ...all }))
      })
      .catch((e) => setError(`Could not load data.json (${e.message}). Run export_ui_data.py first.`))
    if (STATIC_DEMO) {   // no backend to find: go straight to the saved-results mode
      setServer({ state: 'offline' })
      return undefined
    }
    // Poll so the page notices when the backend or the local model is started later.
    const check = () =>
      fetch('api/health')
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.status))))
        .then((h) => setServer({ state: 'live', ...h }))
        // Keep the last known library: a brief outage (e.g. a restart) must not swap the active document.
        .catch(() => setServer({ state: 'offline' }))
    check()
    const timer = setInterval(check, 15000)
    return () => clearInterval(timer)
  }, [])

  const refreshDocs = useCallback(() =>
    fetch('api/documents')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.status))))
      .then((list) => {
        setDocs(list)
        const want = activateWhenReady.current
        if (want && list.some((d) => d.id === want && d.status === 'ready')) {
          activateWhenReady.current = null
          setActiveId(want)
        }
        return list
      })
      .catch(() => null), [setActiveId])

  // Poll the library quickly while something is indexing, slowly otherwise.
  const busyIndexing = docs?.some((d) => d.status === 'queued' || d.status === 'indexing')
  useEffect(() => {
    if (server.state !== 'live') return
    refreshDocs()
    const timer = setInterval(refreshDocs, busyIndexing ? 1200 : 15000)
    return () => clearInterval(timer)
  }, [server.state, busyIndexing, refreshDocs])

  // Before the server has ever answered, only the built-in case study exists (from the saved results).
  const library = docs ?? (data ? [{
    id: GDPR_ID, title: data.document.title, filename: 'gdpr.pdf', builtin: true, status: 'ready',
    pages: data.document.pages, chunks: data.document.chunks, offline: true,
  }] : [])
  const activeDoc = library.find((d) => d.id === activeId) ?? library.find((d) => d.id === GDPR_ID) ?? null

  // The active document's case study: the saved notebook run for the GDPR (plus any questions added from its
  // failure hunt), a generated one for uploads.
  const studyId = activeDoc?.id
  const studyEntry = studyId ? studies[studyId] : null
  const added = studyId === GDPR_ID ? studyEntry?.added : null
  const study = useMemo(() => {
    if (studyId !== GDPR_ID) return studyEntry?.study ?? null
    return data && added?.length ? { ...data, queries: [...data.queries, ...added] } : data
  }, [studyId, studyEntry, data, added])
  const studyState = studyId === GDPR_ID ? { state: 'ready' } : studyEntry ?? { state: 'unknown' }
  const qi = Math.min(qis[studyId] ?? 0, Math.max(0, (study?.queries.length ?? 1) - 1))
  const setQi = useCallback((i) => setQis((all) => ({ ...all, [studyId]: i })), [studyId])

  const fetchStudy = useCallback((id) =>
    fetch(`api/documents/${id}/study`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.status))))
      .then((s) => setStudies((all) => ({ ...all, [id]: s })))
      .catch(() => {}), [])

  // Load the active document's study when it becomes active; poll while it is being built.
  const studyBuilding = studyEntry?.state === 'building'
  const docReady = activeDoc?.status === 'ready'
  useEffect(() => {
    if (!studyId || server.state !== 'live' || !docReady) return
    fetchStudy(studyId)
    if (!studyBuilding) return
    const timer = setInterval(() => fetchStudy(studyId), 1500)
    return () => clearInterval(timer)
  }, [studyId, server.state, docReady, studyBuilding, fetchStudy])

  const buildStudy = useCallback(async (backend) => {
    const id = studyId
    const r = await fetch(`api/documents/${id}/study`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ backend }),
    })
    const body = await r.json().catch(() => ({}))
    if (!r.ok) throw new Error(body.detail ?? `HTTP ${r.status}`)
    setStudies((all) => ({ ...all, [id]: { ...all[id], ...body } }))
  }, [studyId])

  const upload = useCallback((file) => new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', 'api/documents')
    xhr.upload.onprogress = (e) => e.lengthComputable && setUploading({ name: file.name, pct: e.loaded / e.total })
    xhr.onload = () => {
      setUploading(null)
      let body = {}
      try { body = JSON.parse(xhr.responseText) } catch { /* non-JSON error page */ }
      if (xhr.status >= 200 && xhr.status < 300) {
        activateWhenReady.current = body.id
        refreshDocs()
        resolve(body)
      } else reject(new Error(body.detail ?? `Upload failed (HTTP ${xhr.status})`))
    }
    xhr.onerror = () => { setUploading(null); reject(new Error('Upload failed: the server is not reachable.')) }
    const form = new FormData()
    form.append('file', file)
    setUploading({ name: file.name, pct: 0 })
    xhr.send(form)
  }), [refreshDocs])

  const removeDoc = useCallback(async (id) => {
    const r = await fetch(`api/documents/${id}`, { method: 'DELETE' })
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail ?? `HTTP ${r.status}`)
    setTurns((ts) => ts.filter((t) => t.docId !== id))
    setStudies((all) => { const { [id]: gone, ...rest } = all; void gone; return rest })
    if (activeId === id) setActiveId(GDPR_ID)
    refreshDocs()
  }, [activeId, refreshDocs, setActiveId])

  const patch = (id, change) => setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, ...change } : t)))

  /** Merges answer fields into a turn without dropping answers it already has. */
  const patchAnswers = (id, partial) => setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, answers: { ...(t.answers ?? {}), ...partial } } : t)))

  /**
   * Streams the requested answers into a turn. Text is buffered and flushed to state at most every 50 ms,
   * so long answers do not re-render the markdown on every token.
   */
  const streamAnswers = useCallback(async ({ id, query, docId, candidates, backend, pipelines }) => {
    const answers = {}
    let dirty = false
    const flush = () => {
      if (!dirty) return
      dirty = false
      patchAnswers(id, Object.fromEntries(Object.entries(answers).map(([k, v]) => [k, { ...v }])))
    }
    const ticker = setInterval(flush, 50)
    const fail = (status, message) => patch(id, {
      pending: [],
      status: 'done',
      note: status === 429
        ? 'The Gemini free-tier quota is used up for today. Answers that finished are shown; otherwise only the chunks.'
        : status === 503
          ? (server.localHosted
            ? `The hosted open model did not answer (${message}).`
            : 'The local model is not running. Start it in WSL with: bash start_local_llm.sh')
          : `The LLM call failed (${message}).`,
    })
    patch(id, { status: 'answering', backend, note: null, pending: pipelines })
    try {
      const r = await fetch('api/answer/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          doc: docId,
          baseline: top3(candidates, 'biRank').map((c) => c.chunk),
          reranked: top3(candidates, 'ceRank').map((c) => c.chunk),
          backend,
          pipelines,
        }),
      })
      if (!r.ok || !r.body) throw Object.assign(new Error(`HTTP ${r.status}`), { status: r.status })
      const reader = r.body.pipeThrough(new TextDecoderStream()).getReader()
      let buf = ''
      for (;;) {
        const { value, done } = await reader.read()
        if (done) break
        buf += value
        let nl
        while ((nl = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, nl).trim()
          buf = buf.slice(nl + 1)
          if (!line) continue
          const ev = JSON.parse(line)
          if (ev.type === 'start') answers[ev.pipeline] = { text: '', model: ev.model, streaming: true }
          else if (ev.type === 'delta') answers[ev.pipeline].text += ev.text
          else if (ev.type === 'end') Object.assign(answers[ev.pipeline], { streaming: false, ms: ev.ms, firstTokenMs: ev.firstTokenMs })
          else if (ev.type === 'error') { clearInterval(ticker); dirty = true; flush(); fail(ev.status, ev.message); return }
          dirty = true
        }
      }
      clearInterval(ticker)
      dirty = true
      flush()
      patch(id, { status: 'done', pending: [] })
    } catch (e) {
      clearInterval(ticker)
      dirty = true
      flush()
      fail(e.status, e.message)
    }
  }, [server.localHosted])

  /** Generates one more answer (usually the baseline) for a question that is already on screen. */
  const answerMore = useCallback((turn, pipeline) => streamAnswers({
    id: turn.id, query: turn.query, docId: turn.docId, candidates: turn.candidates,
    backend: turn.backend ?? 'gemini', pipelines: [pipeline],
  }), [streamAnswers])

  const ask = useCallback(async (query, { withLlm, backend = 'gemini', compare = false }) => {
    const id = nextId.current++
    const docId = activeDoc?.id ?? GDPR_ID
    const isCase = docId === GDPR_ID
    const test = study ? study.queries.findIndex((q) => sameText(q.query, query)) : -1
    setTurns((ts) => [...ts, { id, docId, query, status: 'retrieving', testIndex: test >= 0 ? test : null }])

    // Stage 1 + 2: live retrieval, or the precomputed result for a known test question.
    let candidates, timings, source
    try {
      const r = await post('api/retrieve', { query, doc: docId })
      candidates = r.candidates
      timings = r.timings
      source = 'live'
    } catch (e) {
      if (test < 0 || !isCase) {
        patch(id, {
          status: 'error',
          error: e.status === 409
            ? 'This document is still being indexed. Try again in a moment.'
            : isCase
              ? (STATIC_DEMO
                ? 'This hosted demo answers only the 13 test questions, from saved results.'
                : 'The live server is offline, so only the 13 test questions can be answered. Start it with: python -m uvicorn server:app --port 8000')
              : `Retrieval failed (${e.message}).`,
        })
        return
      }
      candidates = study.queries[test].candidates
      timings = null
      source = 'saved'
    }
    if (test >= 0) {
      const gold = new Set(study.queries[test].gold)
      candidates = candidates.map((c) => ({ ...c, gold: gold.has(c.chunk) }))
    }
    patch(id, { candidates, timings, source, backend, status: withLlm ? 'answering' : 'done' })
    if (!withLlm) return

    // Stage 3: answers. Saved case-study answers are reused for test questions when they came from the
    // same kind of model, to spare the API quota; otherwise answers are generated fresh.
    const savedBackend = study?.generated?.backend ?? 'gemini'
    const saved = test >= 0 && backend === savedBackend ? study.queries[test].answers : null
    if (saved) {
      patch(id, {
        status: 'done',
        answers: {
          baseline: { text: saved.baseline, model: study.models.llm, saved: true },
          reranked: { text: saved.reranked, model: study.models.llm, saved: true },
        },
      })
      return
    }
    if (source !== 'live') {
      patch(id, { status: 'done', note: 'Answers were not generated for this question in the notebook run.' })
      return
    }
    await streamAnswers({ id, query, docId, candidates, backend, pipelines: compare ? ['reranked', 'baseline'] : ['reranked'] })
  }, [study, activeDoc, streamAnswers])

  const value = {
    data, error, qi, setQi, server, study, studyState, buildStudy, refreshStudy: () => fetchStudy(studyId),
    library, activeDoc,
    // False while the first server check is pending and the remembered document is an upload, so the page
    // waits instead of showing the GDPR and then switching.
    libraryReady: docs != null || server.state === 'offline' || activeId === GDPR_ID, setActiveId, upload, uploading, removeDoc, refreshDocs,
    turns, ask, answerMore, clearChat: (docId) => setTurns((ts) => ts.filter((t) => t.docId !== docId)),
    flowTarget, setFlowTarget,
  }
  return <Store.Provider value={value}>{children}</Store.Provider>
}
