import { useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { GDPR_ID, useStore } from '../store.jsx'
import ReRankBoard from '../components/ReRankBoard.jsx'
import { DEMO_NOTE, STATIC_DEMO } from '../lib.js'

const STAMP = { ready: 'Indexed', indexing: 'Indexing', queued: 'Queued', error: 'Failed' }
const STAGE = { queued: 'waiting for its turn', reading: 'reading and cleaning pages', embedding: 'embedding chunks', ready: 'ready' }

function Dropzone() {
  const { upload, uploading, server } = useStore()
  const [over, setOver] = useState(false)
  const [error, setError] = useState(null)
  const input = useRef(null)
  const live = server.state === 'live'
  const maxMb = server.maxUploadMb ?? 50

  const take = async (files) => {
    setError(null)
    const file = [...files].find((f) => f.type === 'application/pdf' || /\.pdf$/i.test(f.name))
    if (!file) { setError('That is not a PDF. Only PDF files can be added.'); return }
    if (file.size > maxMb * 1024 * 1024) { setError(`That file is ${(file.size / 2 ** 20).toFixed(0)} MB; the limit is ${maxMb} MB.`); return }
    try { await upload(file) } catch (e) { setError(e.message) }
  }

  return (
    <div
      className={`sheet dropzone ${over ? 'over' : ''} ${live ? '' : 'disabled'}`}
      onDragOver={(e) => { e.preventDefault(); if (live) setOver(true) }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => { e.preventDefault(); setOver(false); if (live) take(e.dataTransfer.files) }}
    >
      {uploading ? (
        <div className="dz-busy">
          <span className="sheet-label">Uploading</span>
          <b className="dz-name">{uploading.name}</b>
          <div className="meter big"><i style={{ width: `${Math.round(uploading.pct * 100)}%` }} /></div>
          <span className="mono dz-pct">{Math.round(uploading.pct * 100)}%</span>
        </div>
      ) : (
        <>
          <span className="dz-glyph" aria-hidden="true">PDF</span>
          <p className="dz-title">{live ? 'Drop a PDF to index it' : STATIC_DEMO ? 'Uploading PDFs' : 'Start the server to add PDFs'}</p>
          <p className="dz-sub">
            {live
              ? `Any text-based PDF up to ${maxMb} MB: a policy, a contract, a manual, a rulebook, a paper.`
              : STATIC_DEMO ? DEMO_NOTE : 'Run .venv/Scripts/python -m uvicorn server:app --port 8000 in the project folder, then reload.'}
          </p>
          <input ref={input} id="pdf-input" type="file" accept="application/pdf,.pdf" hidden onChange={(e) => { take(e.target.files); e.target.value = '' }} />
          <button className="btn hot" disabled={!live} onClick={() => input.current?.click()}>Choose a PDF</button>
        </>
      )}
      {error && <p className="red-note small">{error}</p>}
    </div>
  )
}

function DocCard({ doc, active }) {
  const { setActiveId, removeDoc } = useStore()
  const navigate = useNavigate()
  const [confirm, setConfirm] = useState(false)
  const [err, setErr] = useState(null)
  const ready = doc.status === 'ready'
  const open = (to) => { setActiveId(doc.id); navigate(to) }

  return (
    <article className={`sheet doc-card ${active ? 'active' : ''}`}>
      {active && <span className="reading-now">Active</span>}
      <div className="doc-top">
        <span className={`stamp-sm ${doc.status}`}>{STAMP[doc.status] ?? doc.status}</span>
        {(doc.builtin || doc.study === 'ready') && <span className="tag">case study</span>}
      </div>
      <h3>{doc.title}</h3>
      <p className="doc-file mono">{doc.filename}</p>
      <p className="doc-stats mono">
        {doc.pages != null && <span><b>{doc.pages}</b> pages</span>}
        {doc.chunks != null && <span><b>{doc.chunks}</b> chunks</span>}
        {doc.created && <span>added {new Date(doc.created).toLocaleDateString(undefined, { day: 'numeric', month: 'short' })}</span>}
      </p>
      {(doc.status === 'indexing' || doc.status === 'queued') && (
        <div className="doc-progress">
          <div className="meter"><i style={{ width: `${Math.round((doc.progress ?? 0) * 100)}%` }} /></div>
          <span className="mono">{STAGE[doc.stage] ?? doc.stage} · {Math.round((doc.progress ?? 0) * 100)}%</span>
        </div>
      )}
      {doc.status === 'error' && <p className="red-note small">{doc.error}</p>}
      {err && <p className="red-note small">{err}</p>}
      <div className="doc-actions">
        {ready && <button className="btn hot" onClick={() => open('/chat')}>Ask</button>}
        {ready && !doc.offline && <button className="btn" onClick={() => open('/flow')}>Chunk flow</button>}
        {doc.builtin && <Link className="btn" to="/study" onClick={() => setActiveId(doc.id)}>Case study →</Link>}
        {!doc.builtin && ready && (
          <button className="btn" onClick={() => open('/study')}>{doc.study === 'ready' ? 'Case study →' : 'Make a case study'}</button>
        )}
        {!doc.builtin && !confirm && <button className="btn ghost" onClick={() => setConfirm(true)}>Delete</button>}
        {!doc.builtin && confirm && (
          <span className="confirm">
            Delete this PDF and its index?
            <button className="btn danger" onClick={() => removeDoc(doc.id).catch((e) => { setErr(e.message); setConfirm(false) })}>Delete</button>
            <button className="btn ghost" onClick={() => setConfirm(false)}>Keep</button>
          </span>
        )}
      </div>
    </article>
  )
}

export default function Library() {
  const { library, activeDoc, server, data } = useStore()
  const demo = data.queries.find((q) => q.query.startsWith('What is the maximum fine')) ?? data.queries[0]
  const sorted = [...library].sort((a, b) => (a.id === GDPR_ID) - (b.id === GDPR_ID) || (b.created ?? '').localeCompare(a.created ?? ''))
  const { chunkWords, overlapWords } = data.document

  return (
    <>
      <section className="hero wrap lib-hero" aria-labelledby="lib-title">
        <div className="hero-grid">
          <div>
            <h1 id="lib-title"><span className="ln">Any PDF.</span> <span className="ln"><span className="c-bi">Ten</span> passages in,</span> <span className="ln"><span className="c-ce">three</span> out.</span></h1>
            <p className="lede">
              Add a document and ask it questions. Each question pulls <b>10 candidate passages</b> with a fast
              bi-encoder, optionally fused with BM25 keyword search, then a <b>cross-encoder re-reads</b> them to
              choose the 3 the language model sees: MiniLM-L6 by default, or the slower, more accurate MiniLM-L12.
              You get both answers side by side.
            </p>
            <ol className="lib-steps">
              <li><span><b>Read</b> pages and drop running headers, footers and page numbers</span></li>
              <li><span><b>Cut</b> into {chunkWords}-word chunks that overlap by {overlapWords} words</span></li>
              <li><span><b>Embed</b> every chunk once, so searching takes milliseconds</span></li>
              <li><span><b>Ask</b>: retrieve 10, re-rank, answer from 3</span></li>
            </ol>
          </div>
          <Dropzone />
        </div>
      </section>

      <section className="wrap board-band" aria-label="Re-ranking demo">
        <ReRankBoard query={demo} />
      </section>

      <section className="part wrap" style={{ paddingTop: 40 }}>
        <div className="part-head">
          <h2>Library</h2>
          <p>
            {server.state === 'live'
              ? 'Pick a document to ask questions or follow its chunks. Uploaded PDFs are kept in the uploads folder and re-indexed when the server restarts.'
              : STATIC_DEMO
                ? 'This hosted demo shows the GDPR case study from saved results.'
                : 'The server is offline, so only the GDPR case study is available, from saved results.'}
          </p>
        </div>
        <div className="shelf">
          {sorted.map((d) => <DocCard key={d.id} doc={d} active={d.id === activeDoc?.id} />)}
        </div>
      </section>
    </>
  )
}
