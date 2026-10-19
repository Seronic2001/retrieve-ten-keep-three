import { useState } from 'react'
import { fmtDuration, fmtInt } from '../lib.js'

const PRESETS = [100_000, 1_000_000, 10_000_000]

function Receipt({ title, lines, withLlm, note, runs }) {
  const total = lines.filter((l) => withLlm || !l.llm).reduce((s, l) => s + l.ms, 0)
  return (
    <div className="sheet receipt">
      <h3>{title}</h3>
      <p className="rmeta">1 question · {runs ? `mean over ${runs} test questions` : 'mean of 5 runs'}</p>
      {lines.map((l) => (
        <div key={l.name} className={`line ${l.add ? 'add' : ''} ${l.llm && !withLlm ? 'off' : ''}`}>
          <span>{l.name}</span><span className="dots" /><span>{fmtDuration(l.ms)}</span>
        </div>
      ))}
      <div className="total"><span>TOTAL</span><span>{fmtDuration(total)}</span></div>
      {note && <p className="receipt-note">{note}</p>}
    </div>
  )
}

export default function Cost({ data }) {
  const { retrieveMs, rerankMs, llmMs, runs } = data.latency
  const hasLlm = llmMs != null   // a generated study whose answers hit the quota has no LLM timing
  const { perPairMs, corpus } = data.scaling
  const device = data.models.device?.startsWith('cuda') ? 'a CUDA GPU' : 'CPU'
  const [withLlm, setWithLlm] = useState(false)

  const lo = Math.log10(corpus), hi = 7
  const [pos, setPos] = useState(1000) // slider position 0..1000 on a log scale; starts at 10M
  const n = Math.max(corpus, Math.round(10 ** (lo + ((hi - lo) * pos) / 1000)))
  const setN = (v) => setPos(Math.round(((Math.log10(v) - lo) / (hi - lo)) * 1000))

  const fullScan = n * perPairMs
  const twoStage = retrieveMs + rerankMs
  const base = [
    { name: 'Embed question + FAISS top 10', ms: retrieveMs },
    ...(hasLlm ? [{ name: `LLM answer (${data.models.llm})`, ms: llmMs, llm: true }] : []),
  ]
  const overhead = withLlm && hasLlm
    ? `+${((rerankMs / (retrieveMs + llmMs)) * 100).toFixed(1)}% end to end`
    : `${((retrieveMs + rerankMs) / retrieveMs).toFixed(1)}× slower retrieval`

  return (
    <section id="cost" className="part wrap">
      <div className="part-head">
        <h2>Ten pairs are cheap. Every pair is not.</h2>
        <p>Timed on {device} after a warm-up run. Toggle the LLM line to see how much of the total re-ranking really costs.</p>
      </div>

      <div className="receipts">
        <Receipt title="Pipeline 1 · Baseline" lines={base} withLlm={withLlm} runs={runs} />
        <Receipt
          title="Pipeline 2 · Re-ranked"
          lines={[base[0], { name: 'Cross-encoder × 10 pairs', ms: rerankMs, add: true }, ...base.slice(1)]}
          withLlm={withLlm}
          runs={runs}
          note={overhead}
        />
      </div>
      <div className="cost-bar">
        <div className="tabs" role="group" aria-label="Include LLM time">
          <button aria-pressed={!withLlm} onClick={() => setWithLlm(false)}>Retrieval only</button>
          <button aria-pressed={withLlm} disabled={!hasLlm} onClick={() => setWithLlm(true)}>Include the LLM</button>
        </div>
        <span className="hand">{withLlm ? 'the LLM dwarfs everything else' : 'zoomed in on retrieval'}</span>
      </div>

      <div className="scale">
        <h3>What if the cross-encoder read every chunk?</h3>
        <p className="sub">
          Each (question, chunk) pair needs its own forward pass: {fmtDuration(perPairMs)} here, and nothing can be
          precomputed. Bi-encoder chunk vectors are built once, offline. Drag to grow the corpus.
        </p>
        <div className="scale-ctrl">
          <label htmlFor="corpus">Corpus: <b className="mono">{fmtInt(n)}</b> chunks</label>
          <input id="corpus" type="range" min={0} max={1000} value={pos} onChange={(e) => setPos(+e.target.value)} />
          <div className="presets">
            {[corpus, ...PRESETS.filter((p) => p > corpus)].map((p) => (
              <button key={p} className="btn" onClick={() => setN(p)}>{p === corpus ? `This PDF (${fmtInt(corpus)})` : fmtInt(p)}</button>
            ))}
          </div>
        </div>
        <div className="duel">
          <div className="full">
            <span className="k">Cross-encoder over everything</span>
            <span className="v">{fmtDuration(fullScan)}</span>
            <span className="x">{fmtInt(n)} forward passes per question</span>
          </div>
          <div className="two">
            <span className="k">Retrieve 10, re-rank 10</span>
            <span className="v">{fmtDuration(twoStage)}</span>
            <span className="x">one vector search + 10 forward passes</span>
          </div>
        </div>
        <p className="ratio-note">
          → {fmtInt(fullScan / twoStage)}× slower per question. (Search time as measured here; an ANN index such as
          HNSW keeps it in milliseconds at scale.)
        </p>
      </div>
    </section>
  )
}
