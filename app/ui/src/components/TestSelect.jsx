import { useStore } from '../store.jsx'
import { VERDICT, rankText } from '../lib.js'

/** Compact picker for the case study's test questions, shared by the single-question pages. */
export default function TestSelect() {
  const { study, qi, setQi } = useStore()
  const q = study.queries[qi]
  return (
    <div className="test-select">
      <label htmlFor="test-q" className="hand">Test question</label>
      <select id="test-q" value={qi} onChange={(e) => setQi(+e.target.value)}>
        {study.queries.map((t, i) => (
          <option key={t.query} value={i}>{t.query}</option>
        ))}
      </select>
      <span className={`verdict-word ${q.status}`}>
        {rankText(q.baselineRank)} → {rankText(q.rerankedRank)} · {VERDICT[q.status]}
      </span>
    </div>
  )
}
