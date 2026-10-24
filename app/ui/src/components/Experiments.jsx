import { useStore } from '../store.jsx'
import { fmtDuration } from '../lib.js'

const DOCS = [['gdpr', 'GDPR'], ['constitution', 'Constitution of India']]
const PIPELINES = {
  'dense+MiniLM-L6': 'Dense + MiniLM-L6 (Pipeline 2)',
  'hybrid+MiniLM-L6': 'Hybrid + MiniLM-L6',
  'hybrid+bge-v2-m3': 'Hybrid + bge-reranker-v2-m3',
}
const pc = (x) => `${Math.round(x * 100)}%`

/** Follow-up experiments on both test documents: hybrid retrieval, stronger re-rankers, judged answer quality. */
export default function Experiments() {
  const { data } = useStore()
  const ex = data.experiments
  if (!ex) {
    return (
      <section className="part wrap">
        <p className="offline sheet sheet-pad">
          These results are missing from data.json. Re-run export_ui_data.py from notebook/, then reload.
        </p>
      </section>
    )
  }
  const fastest = ex.rerankers.find((r) => r.ms > 0)

  return (
    <section id="experiments" className="part wrap">
      <div className="part-head">
        <h2>What we tried next</h2>
        <p>
          The case study compares the notebook&apos;s two pipelines. After it, both test documents were used to try
          stronger retrieval: <b>hybrid search</b> (the bi-encoder&apos;s and BM25 keyword search&apos;s top 20, fused by
          Reciprocal Rank Fusion), <b>larger re-rankers</b>, and a check of the <b>answers</b> themselves, not only the
          chunks. Questions: the GDPR&apos;s 13 hand-labelled and 100 failure-hunt questions, and the Constitution&apos;s
          case-study and failure-hunt questions with their gold corrected by hand ({ex.questions.gdpr} and{' '}
          {ex.questions.constitution}).
        </p>
      </div>

      <h3 className="exp-h">Re-rankers on a CPU, behind hybrid retrieval</h3>
      <div className="sheet ledger hunt-tally">
        <table>
          <thead>
            <tr>
              <th>Re-ranker</th>
              {DOCS.map(([k, name]) => <th key={k}>{name} · answer in top 3</th>)}
              <th>Time per question · {ex.cpuThreads} threads</th>
              <th>· 2 threads</th>
            </tr>
          </thead>
          <tbody>
            {ex.rerankers.map((r) => (
              <tr key={r.name} className={r === fastest ? 'hl-row' : ''}>
                <td className="cat">{r.name}</td>
                {DOCS.map(([k]) => <td key={k} className="mono"><b>{pc(r[k])}</b></td>)}
                <td className="mono">{r.ms ? fmtDuration(r.ms) : '–'}</td>
                <td className="mono">{r.ms2 ? fmtDuration(r.ms2) : '–'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="side-note exp-note">
        Each re-ranker scores the same 10 fused candidates with ONNX Runtime, as the app does. bge-reranker-v2-m3 finds
        the answer most often but is far too slow without a GPU, so the chat offers MiniLM-L6 (the default, fastest) and
        MiniLM-L12, which gains 3–6 points for about twice the time. Compressing the models to int8 kept their accuracy
        but barely sped them up on this CPU.
      </p>

      <h3 className="exp-h">Are the answers right?</h3>
      <div className="sheet ledger hunt-tally">
        <table>
          <thead>
            <tr>
              <th>Pipeline</th>
              <th>Document</th>
              <th>Correct</th>
              <th>Partly</th>
              <th>Wrong</th>
              <th>Declined</th>
              <th>Answer in top 3</th>
            </tr>
          </thead>
          <tbody>
            {DOCS.flatMap(([doc, name]) => ex.answers.filter((a) => a.doc === doc).map((a) => (
              <tr key={`${doc}/${a.pipeline}`}>
                <td className="cat">{PIPELINES[a.pipeline] ?? a.pipeline}</td>
                <td>{name}</td>
                <td className="mono"><b>{pc(a.correct)}</b></td>
                <td className="mono">{pc(a.partial)}</td>
                <td className="mono">{pc(a.wrong)}</td>
                <td className="mono">{pc(a.abstained)}</td>
                <td className="mono">{pc(a.goldInTop3)}</td>
              </tr>
            )))}
          </tbody>
        </table>
      </div>
      <p className="side-note exp-note">
        Answers by Qwen3.5-4B from each pipeline&apos;s top 3, graded by a larger model (Qwen3-14B) against the answer
        passage; a first pass that does not see the answer decides whether the model declined. Its labels agree with
        87% of 60 hand-graded answers. Hybrid retrieval cuts the declined answers most; wrong answers stay rare
        (4–9%). Checking the Constitution&apos;s wrong answers by hand found that most were bad test labels, now corrected.
      </p>
    </section>
  )
}
