import { CATEGORY, VERDICT, rankText } from '../lib.js'

// Slots 1..10 plus one for "not retrieved"; the top-3 slots are tinted.
export function RankTrack({ from, to, status }) {
  const slot = 17, pad = 6, W = pad * 2 + slot * 11, H = 42
  const x = (r) => pad + ((r ?? 11) - 1) * slot + slot / 2
  const y = 27
  const a = x(from), b = x(to)
  return (
    <svg className="track" width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img"
      aria-label={`bi-encoder ${rankText(from)}, cross-encoder ${rankText(to)}`}>
      {Array.from({ length: 11 }, (_, i) => (
        <g key={i}>
          <rect className={`slot ${i < 3 ? 'top' : ''}`} x={pad + i * slot + 1} y={y - 6} width={slot - 2} height={12} rx={2} />
          {(i === 0 || i === 2 || i === 9 || i === 10) && (
            <text className="lab" x={pad + i * slot + slot / 2} y={H - 1} textAnchor="middle">{i === 10 ? '>10' : i + 1}</text>
          )}
        </g>
      ))}
      {a !== b && <path className={`arc ${status}`} d={`M${a},${y - 7} Q${(a + b) / 2},${-6} ${b},${y - 7}`} />}
      <circle className="from" cx={a} cy={y} r={4.5} />
      <circle className={`to ${status}`} cx={b} cy={y} r={a === b ? 2.5 : 4.5} />
    </svg>
  )
}

// One cell per generated question, like a dot-matrix board: rescued, broken, unchanged.
export function Waffle({ restored, degraded, total }) {
  const cells = Array.from({ length: total }, (_, i) => (i < restored ? 'r' : i < restored + degraded ? 'd' : 'u'))
  return (
    <div className="waffle" role="img" aria-label={`${restored} rescued, ${degraded} broken, ${total - restored - degraded} unchanged`}>
      {cells.map((c, i) => <i key={i} className={c} />)}
    </div>
  )
}

export default function Scoreboard({ data, qi, onPick }) {
  const { queries, sweep } = data
  const counts = queries.reduce((m, q) => ({ ...m, [q.status]: (m[q.status] ?? 0) + 1 }), {})
  const unchanged = sweep ? sweep.questions - sweep.restored - sweep.degraded : 0

  return (
    <section id="scoreboard" className="part wrap">
      <div className="part-head">
        <h2>
          {counts.restored} rescued, {counts.degraded ?? 0} broken
        </h2>
        <p>
          Where the answer chunk sat for every test question. Hollow dot: bi-encoder rank. Filled dot: rank after
          re-ranking. The shaded slots are the top 3. Click a row to replay it.
        </p>
      </div>

      <div className="sheet ledger">
        <table>
          <thead>
            <tr><th>Failure type</th><th>Question</th><th>Bi → cross rank</th><th>Result</th></tr>
          </thead>
          <tbody>
            {queries.map((q, i) => (
              <tr key={q.query} aria-selected={i === qi} tabIndex={0}
                onClick={() => onPick(i, 'replay')}
                onKeyDown={(e) => e.key === 'Enter' && onPick(i, 'replay')}>
                <td className="cat">{CATEGORY[q.category]}</td>
                <td className="qtext">{q.query}</td>
                <td><RankTrack from={q.baselineRank} to={q.rerankedRank} status={q.status} /></td>
                <td><span className={`verdict-word ${q.status}`}>{VERDICT[q.status]}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {sweep && <div className="tally">
        <div className="tally-copy">
          <h3>Stress test: {sweep.questions} questions</h3>
          <p>
            Gemini wrote two questions for every chunk. Each was run through both pipelines with its source chunk as
            the answer, to see how often re-ranking helps or hurts beyond the hand-picked cases.
          </p>
        </div>
        <div className="sheet sheet-pad tally-sheet">
          <div className="tally-keys">
            <span className="tally-row r"><b>{sweep.restored}</b> rescued</span>
            <span className="tally-row d"><b>{sweep.degraded}</b> broken</span>
            <span className="tally-row u"><b>{unchanged}</b> unchanged</span>
          </div>
          <Waffle restored={sweep.restored} degraded={sweep.degraded} total={sweep.questions} />
          <p className="tally-rest">
            “Broken” means the source chunk was in the baseline top 3, and neither it nor
            an overlapping neighbour made the re-ranked top 3.
          </p>
        </div>
      </div>}
    </section>
  )
}
