import { useEffect, useRef, useState } from 'react'
import { byBi, byCe, rankText } from '../lib.js'

const steps = (chunks) => [
  { n: String(chunks), label: 'chunks on file' },
  { n: '10', label: 'bi-encoder shortlist' },
  { n: '3', label: 'baseline keeps' },
  { n: '3', label: 're-ranker keeps' },
]

// Stable pseudo-random word widths so the greeked page looks like set text.
const widthOf = (id) => 14 + ((id * 2654435761) >>> 0) % 38

export default function Hero({ data, query }) {
  const [step, setStep] = useState(3)
  const timer = useRef(null)
  const doc = data.document
  const isGdpr = !data.generated   // the saved notebook run; generated studies describe their own document
  const STEPS = steps(doc.chunks)

  const shortlist = new Set(query.candidates.map((c) => c.chunk))
  const kept = new Set((step === 2 ? byBi(query) : step === 3 ? byCe(query) : []).slice(0, 3).map((c) => c.chunk))
  const gold = new Set(query.gold)
  const goldKept = [...gold].some((g) => kept.has(g))

  useEffect(() => () => clearInterval(timer.current), [])

  const play = () => {
    clearInterval(timer.current)
    let s = 0
    setStep(0)
    timer.current = setInterval(() => {
      s += 1
      setStep(s)
      if (s >= 3) clearInterval(timer.current)
    }, 1200)
  }

  const word = (id) => {
    const cls = ['w']
    if (step >= 1) cls.push(shortlist.has(id) ? 'short' : 'dim')
    if (gold.has(id)) cls.push('gold')
    if (kept.has(id)) cls.push('keep')
    return <i key={id} className={cls.join(' ')} style={{ width: widthOf(id) }} title={`chunk ${id}`} />
  }
  const half = Math.ceil(doc.chunks / 2)
  const ids = Array.from({ length: doc.chunks }, (_, i) => i)
  const column = (from, to) => (
    <div className="oj-col">
      {ids.slice(from, to).flatMap((id) =>
        isGdpr && id === doc.firstArticleChunk
          ? [<span key="div" className="oj-divider">HAVE ADOPTED THIS REGULATION:</span>, word(id)]
          : [word(id)],
      )}
    </div>
  )

  const captions = [
    <>{isGdpr ? 'The regulation' : 'The document'} as {doc.chunks} overlapping {doc.chunkWords}-word chunks. The <span className="hl">highlighted</span> one answers the question below.</>,
    <>The bi-encoder keeps the 10 chunks whose vectors sit closest to the question. The answer chunk is at <b>{rankText(query.baselineRank)}</b>.</>,
    <>Pipeline 1 keeps its top 3 and hands them to the LLM. The answer is <b>{goldKept ? 'inside' : 'not inside'}</b> that context.</>,
    <>Pipeline 2 re-reads the same 10 with a cross-encoder. The answer moves to <b>{rankText(query.rerankedRank)}</b> and is <b>{goldKept ? 'inside' : 'not inside'}</b> the context.</>,
  ]

  return (
    <section className="hero wrap" aria-labelledby="hero-title">
      <div className="hero-grid">
        <div>
          <h1 id="hero-title">
            Retrieve <span className="strike">ten</span>
            <br />
            Keep <span className="hl">three</span>
          </h1>
          <p className="lede">
            Two retrieval pipelines answer questions about {isGdpr ? <b>the {doc.pages}-page GDPR</b> : <><b>{doc.title}</b> ({doc.pages} pages)</>}.
            Both pull the same ten candidate passages. They differ only in how they pick the three the language model
            gets to read.
          </p>
          <div className="meta-row">
            <span><b>{doc.pages}</b> pages</span>
            <span><b>{doc.chunks}</b> chunks</span>
            <span><b>{data.queries.length}</b> {isGdpr ? 'test' : 'generated test'} questions</span>
            {isGdpr && <span>Regulation (EU) 2016/679</span>}
          </div>
        </div>

        <div className="sheet ojpage" aria-label="Document map">
          <div className="oj-head">
            {isGdpr
              ? <><span>L 119/1</span><span>Official Journal of the European Union</span><span>EN</span></>
              : <><span className="oj-title">{doc.title}</span><span>{doc.pages} pp.</span></>}
          </div>
          <div className="oj-cols" aria-hidden="true">
            {column(0, half)}
            {column(half, doc.chunks)}
          </div>
          <div className="oj-steps" role="group" aria-label="Pipeline stage">
            {STEPS.map((s, i) => (
              <button key={i} aria-pressed={step === i} onClick={() => { clearInterval(timer.current); setStep(i) }}>
                <b>{s.n}</b>
                {s.label}
              </button>
            ))}
          </div>
          <p className="oj-caption">{captions[step]}</p>
          <div className="oj-foot">
            <span className="q">Q: {query.query}</span>
            <button className="btn hot" onClick={play}>▶ Play the funnel</button>
          </div>
        </div>
      </div>
    </section>
  )
}
