import Markdown from 'react-markdown'
import { VERDICT, byBi, byCe, rankText } from '../lib.js'
import AnswerChecks from './AnswerChecks.jsx'

function AnswerSheet({ title, sub, ctx, text, pipeline }) {
  const found = ctx.some((c) => c.gold)
  return (
    <article className={`sheet sheet-pad ans ${pipeline}`}>
      <div className="ans-title">
        <div>
          <h3>{title}</h3>
          <p className="sub">{sub}</p>
        </div>
        <span className={`stamp ${found ? 'found' : 'missing'}`}>
          {found ? 'Clause in context' : 'Clause missing'}
        </span>
      </div>
      <p className="ctx-line">
        Context:{' '}
        {ctx.map((c, i) => (
          <span key={c.chunk}>
            <b className={c.gold ? 'gold' : ''}>{c.cite}</b>{i < ctx.length - 1 ? ' · ' : ''}
          </span>
        ))}
      </p>
      <div className="ans-body">
        <Markdown>{text}</Markdown>
      </div>
      <AnswerChecks text={text} chunks={ctx} />
    </article>
  )
}

export default function Answers({ query, queries, onPick }) {
  const withAnswers = queries.map((q, i) => [q, i]).filter(([q]) => q.answers)

  return (
    <section id="answers" className="part wrap">
      <div className="part-head">
        <h2>Same prompt. Different three chunks.</h2>
        <p>Both answers come from the same LLM with the same prompt. Only the context changes.</p>
      </div>

      {query.answers ? (
        <>
          <p className="ans-q">{query.query}</p>
          <div className="ans-grid">
            <AnswerSheet
              title="Answer 1 · Baseline"
              sub={`bi-encoder top 3 · answer chunk at ${rankText(query.baselineRank)}`}
              ctx={byBi(query).slice(0, 3)}
              text={query.answers.baseline}
              pipeline="baseline"
            />
            <AnswerSheet
              title="Answer 2 · Re-ranked"
              sub={`cross-encoder top 3 · answer chunk at ${rankText(query.rerankedRank)}`}
              ctx={byCe(query).slice(0, 3)}
              text={query.answers.reranked}
              pipeline="reranked"
            />
          </div>
        </>
      ) : (
        <div className="sheet sheet-pad ans-empty">
          <p>
            Answers were generated only where the two pipelines gave the LLM different context. For “{query.query}”
            the result is <b>{VERDICT[query.status]}</b>, so there is nothing to compare. Try one of these:
          </p>
          <div className="picker" style={{ marginBottom: 0 }}>
            {withAnswers.map(([q, i]) => (
              <button key={q.query} className="slip" onClick={() => onPick(i)}>
                <span className="q">{q.query}</span>
                <span className={`verdict-word ${q.status}`}>{VERDICT[q.status]}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </section>
  )
}
