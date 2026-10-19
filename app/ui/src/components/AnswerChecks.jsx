import { isRefusal, verifyQuotes } from '../lib.js'

/**
 * Checks on a finished answer, against the chunks it was given:
 * - says plainly when the model reports that the passages do not answer the question;
 * - confirms that every quotation in the answer appears word for word in those chunks,
 *   and lists any that do not (a fabricated or misattributed quote).
 */
export default function AnswerChecks({ text, chunks }) {
  if (!text) return null
  const refusal = isRefusal(text)
  const quotes = verifyQuotes(text, chunks)
  const missing = quotes.filter((q) => !q.found)
  if (!refusal && !quotes.length) return null

  return (
    <div className="checks">
      {refusal && (
        <p className="check refusal">
          <span className="check-icon" aria-hidden="true">○</span>
          The model says the 3 passages it read do not directly answer this.
        </p>
      )}
      {quotes.length > 0 && !missing.length && (
        <p className="check ok">
          <span className="check-icon" aria-hidden="true">✓</span>
          {quotes.length === 1 ? 'The quote appears' : `All ${quotes.length} quotes appear`} word for word in the sources.
        </p>
      )}
      {missing.length > 0 && (
        <div className="check bad" role="alert">
          <p>
            <span className="check-icon" aria-hidden="true">!</span>
            {missing.length} of {quotes.length} quote{quotes.length === 1 ? '' : 's'} not found in the sources this answer
            read. Treat {missing.length === 1 ? 'it' : 'them'} as unverified.
          </p>
          <ul>
            {missing.map((q) => <li key={q.quote}>“{q.quote}”</li>)}
          </ul>
        </div>
      )}
    </div>
  )
}
