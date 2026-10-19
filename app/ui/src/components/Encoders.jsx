import { useMemo, useState } from 'react'
import { byCe, sigmoid } from '../lib.js'

const tokens = (s) => s.match(/[\w€%'’-]+|[^\s\w]/g) ?? []
// Space before words, never before punctuation.
const gap = (w, i) => (i > 0 && /^[\w€(‘']/.test(w) ? ' ' : '')
const stem = (w) => w.toLowerCase().replace(/[^a-z0-9]/g, '').slice(0, 5)

// An embedding drawn as a barcode: same text, same stripes.
function Barcode({ text }) {
  let h = 2166136261
  for (const ch of text) h = Math.imul(h ^ ch.charCodeAt(0), 16777619)
  const bars = Array.from({ length: 64 }, () => {
    h = Math.imul(h ^ (h >>> 13), 1274126177)
    return 1 + ((h >>> 0) % 4)
  })
  return (
    <div className="barcode" aria-hidden="true">
      {bars.map((w, i) => <i key={i} style={{ width: w, opacity: i % 3 === 2 ? 0 : 1 }} />)}
    </div>
  )
}

export default function Encoders({ query }) {
  const [hover, setHover] = useState(null)

  // Show the chunk that holds the answer, cut to a short window around the gold snippet.
  const chunk = query.candidates.find((c) => c.gold) ?? byCe(query)[0]
  const excerpt = useMemo(() => {
    const lower = chunk.text.toLowerCase().replace(/[‘’]/g, "'")
    const at = query.goldSnippets.map((s) => lower.indexOf(s)).find((i) => i >= 0) ?? -1
    const startWord = at >= 0 ? chunk.text.slice(0, at).split(' ').length - 1 : 0
    const from = Math.max(0, startWord - 8)
    return chunk.text.split(' ').slice(from, from + 34).join(' ')
  }, [chunk, query])

  const qTok = tokens(query.query)
  const cTok = tokens(excerpt)
  const hotStem = hover != null ? stem(qTok[hover]) : null
  const isMatch = (w) => hotStem && hotStem.length >= 3 && stem(w) === hotStem

  const qWords = (
    <p className="text-q">
      {qTok.map((w, i) => (
        <span key={i}>
          {gap(w, i)}<span
            className={`tw query-word ${hover === i ? 'on' : ''}`}
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
          >{w}</span>
        </span>
      ))}
    </p>
  )

  return (
    <section id="readers" className="part wrap">
      <div className="part-head">
        <h2>One reads apart. One reads together.</h2>
        <p>
          The bi-encoder squeezes the question and each chunk into separate vectors and compares them. The
          cross-encoder reads the question and one chunk as a single sequence, so every word can look at every
          other word.
        </p>
        <span className="margin-note">Hover a word in either question</span>
      </div>

      <div className="enc-grid">
        <article className="sheet sheet-pad enc">
          <div>
            <h3>Bi-encoder</h3>
            <p className="sub">all-MiniLM-L6-v2 · cosine of two 384-d vectors</p>
          </div>
          <div className="textbox">
            <span className="sheet-label">Question</span>
            {qWords}
            <div className="barcode-row"><span>query vector</span><Barcode text={query.query} /></div>
          </div>
          <div className="textbox">
            <span className="sheet-label">Chunk · encoded once, offline</span>
            <p className="text-c">{cTok.map((w, i) => <span key={i}>{gap(w, i)}<span className="tw">{w}</span></span>)}</p>
            <div className="barcode-row"><span>chunk vector</span><Barcode text={excerpt} /></div>
          </div>
          <p className="enc-note">
            {hover == null
              ? 'The two barcodes are all that get compared.'
              : <>“{qTok[hover]}” only changes the query barcode. It never meets the chunk’s words.</>}
          </p>
          <div className="score-stamp">
            <b>{chunk.biScore.toFixed(2)}</b>
            <span>cosine similarity<br />rank #{chunk.biRank} of 10</span>
          </div>
        </article>

        <article className="sheet sheet-pad enc">
          <div>
            <h3>Cross-encoder</h3>
            <p className="sub">ms-marco-MiniLM-L-6-v2 · one score per (question, chunk) pair</p>
          </div>
          <div className="textbox">
            <span className="sheet-label">One joint sequence · full self-attention</span>
            <p className="text-c">
              <span className="tw sep">[CLS]</span>
              {qTok.map((w, i) => (
                <span key={`q${i}`}>
                  {gap(w, 1)}<span
                    className={`tw query-word ${hover === i ? 'on' : ''}`}
                    style={{ fontWeight: 600 }}
                    onMouseEnter={() => setHover(i)}
                    onMouseLeave={() => setHover(null)}
                  >{w}</span>
                </span>
              ))}
              {' '}<span className="tw sep">[SEP]</span>
              {cTok.map((w, i) => (
                <span key={`c${i}`}>
                  {gap(w, 1)}<span className={`tw ${isMatch(w) ? 'match' : hover != null ? 'reach' : ''}`}>{w}</span>
                </span>
              ))}
              {' '}<span className="tw sep">[SEP]</span>
            </p>
          </div>
          <p className="enc-note">
            {hover == null
              ? 'Question and chunk go through the model together.'
              : <>“{qTok[hover]}” can attend to every chunk word, and most strongly to direct matches.</>}
          </p>
          <div className="score-stamp">
            <b>{chunk.ceScore.toFixed(1)}</b>
            <span>relevance logit ({(sigmoid(chunk.ceScore) * 100).toFixed(0)}%)<br />rank #{chunk.ceRank} of 10</span>
          </div>
        </article>
      </div>
      <p className="enc-foot">
        Chunk shown: #{chunk.chunk} · {chunk.cite} · p. {chunk.page}{chunk.gold ? ' · holds the answer' : ''}
      </p>
    </section>
  )
}
