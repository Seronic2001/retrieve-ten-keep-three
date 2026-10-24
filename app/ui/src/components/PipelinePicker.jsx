import { DEFAULT_RERANKER, RERANKERS } from '../store.jsx'

/** Retrieval mode (dense or hybrid) and re-ranker for a live question; both need the server. */
export default function PipelinePicker({ retrieval, setRetrieval, reranker, setReranker, live }) {
  return (
    <>
      <div className="tabs" role="group" aria-label="Retrieval">
        <button type="button" aria-pressed={retrieval === 'dense'} onClick={() => setRetrieval('dense')}
          title="Bi-encoder top 10, as in the notebook">Dense</button>
        <button type="button" aria-pressed={retrieval === 'hybrid'} disabled={!live}
          title="Bi-encoder and BM25 keyword search, fused by Reciprocal Rank Fusion" onClick={() => setRetrieval('hybrid')}>
          Hybrid + BM25
        </button>
      </div>
      <div className="tabs" role="group" aria-label="Re-ranker">
        {Object.entries(RERANKERS).map(([name, title]) => (
          <button key={name} type="button" aria-pressed={reranker === name} title={title}
            disabled={name !== DEFAULT_RERANKER && !live} onClick={() => setReranker(name)}>
            {name}
          </button>
        ))}
      </div>
    </>
  )
}
