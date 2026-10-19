// Built with VITE_STATIC_DEMO=1 for the static Hugging Face Space: there is no backend, only saved results.
export const STATIC_DEMO = import.meta.env.VITE_STATIC_DEMO === '1'
export const DEMO_NOTE = 'Not available in this hosted demo, which runs on saved results.'

export const CATEGORY = {
  keyword_trap: 'Keyword trap',
  exception: 'Exception',
  negation: 'Negation',
  buried: 'Buried fact',
  control: 'Control',
  degraded: 'Re-rank hurts',
  trap: 'Wording trap',
}

export const STATUS = {
  restored: 'Restored',
  degraded: 'Degraded',
  'both-hit': 'Both find it',
  'both-miss': 'Both miss',
}

export const VERDICT = {
  restored: 'restored ✓',
  degraded: 'made worse ✗',
  'both-hit': 'both find it',
  'both-miss': 'both miss it',
}

export const rankText = (r) => (r == null ? '>10' : `#${r}`)

export const byBi = (q) => [...q.candidates].sort((a, b) => a.biRank - b.biRank)
export const byCe = (q) => [...q.candidates].sort((a, b) => a.ceRank - b.ceRank)

// Cross-encoder outputs a logit; squash it to 0..1 for the score bars.
export const sigmoid = (x) => 1 / (1 + Math.exp(-x))

// Straighten curly quotes without changing string length, so indexes still line up.
const straighten = (s) => s.toLowerCase().replace(/[‘’]/g, "'").replace(/[“”]/g, '"')

/** Splits text into [{text, hit}] parts, marking every occurrence of any gold snippet. */
export function markSnippets(text, snippets) {
  const lower = straighten(text)
  const ranges = []
  for (const s of snippets) {
    const needle = straighten(s)
    let i = lower.indexOf(needle)
    while (i !== -1) {
      ranges.push([i, i + needle.length])
      i = lower.indexOf(needle, i + 1)
    }
  }
  ranges.sort((a, b) => a[0] - b[0])
  const parts = []
  let pos = 0
  for (const [a, b] of ranges) {
    if (a < pos) continue
    if (a > pos) parts.push({ text: text.slice(pos, a), hit: false })
    parts.push({ text: text.slice(a, b), hit: true })
    pos = b
  }
  if (pos < text.length) parts.push({ text: text.slice(pos), hit: false })
  return parts
}

export function fmtDuration(ms) {
  if (ms < 1000) return `${ms.toFixed(ms < 10 ? 1 : 0)} ms`
  const s = ms / 1000
  if (s < 60) return `${s.toFixed(1)} s`
  const m = s / 60
  if (m < 60) return `${m.toFixed(1)} min`
  const h = m / 60
  if (h < 48) return `${h.toFixed(1)} h`
  return `${(h / 24).toFixed(1)} days`
}

export const fmtInt = (n) => Math.round(n).toLocaleString('en-US')

// ── answer checks ───────────────────────────────────────────────

// Compare text without case, punctuation or footnote markers ("1[", "]"), which models often tidy up.
const fold = (s) => s.toLowerCase()
  .replace(/\d{1,2}\[/g, ' ')
  .replace(/[^\p{L}\p{N}%€$]+/gu, ' ')
  .replace(/\s+/g, ' ')
  .trim()
const words = (s) => fold(s).split(' ').filter(Boolean)

/**
 * Finds the passages an answer presents as quotations ("…" spans and > blockquotes) and checks each one
 * against the chunks that answer was given. Quotes with an ellipsis or an editorial [insertion] are checked
 * piece by piece, and spaces are ignored so words split by PDF extraction ("dispropor tionate") still match.
 * Short quoted terms (under 4 words, e.g. "purpose limitation") are ignored.
 */
export function verifyQuotes(answer, chunks) {
  if (!answer || !chunks?.length) return []
  const squash = (t) => fold(t).replace(/ /g, '')
  const source = chunks.map((c) => squash(c.text)).join('|')
  const quotes = []
  // Consume every quote pair in order, short ones too, so a closing mark is never paired with the next opening one.
  for (const m of answer.matchAll(/[“"]([^”"\n]{1,600})[”"]/g)) quotes.push(m[1])
  const block = answer.split('\n').filter((l) => /^\s*>/.test(l)).map((l) => l.replace(/^\s*>\s?/, '')).join(' ')
  for (const part of block.split(/\n{2,}/)) if (part.trim()) quotes.push(part.replace(/[*_]/g, ''))
  const seen = new Set()
  return quotes
    .map((q) => q.replace(/\*\*|__/g, '').trim())
    .filter((q) => words(q).length >= 4 && !seen.has(fold(q)) && seen.add(fold(q)))
    .map((q) => {
      const pieces = q.split(/\.\.\.|…|\[[^\]]*\]/).filter((p) => words(p).length >= 3)
      const found = (pieces.length ? pieces : [q]).every((p) => source.includes(squash(p)))
      return { quote: q, found }
    })
}

const REFUSAL = /\b(does not|doesn't|do not|did not) (contain|include|provide|mention|specify|list|state|explicitly)|\bnot (explicitly )?(stated|mentioned|specified|provided|included|listed)\b|\bno (information|mention|details)\b|\bcannot (answer|be answered|determine)|\bnot possible to (answer|determine)/i

/**
 * True when the answer says the passages it was given do not answer the question. Only the opening
 * sentence counts: that is where a refusal is stated, while a real answer may later say "the context does
 * not mention any exceptions".
 */
export function isRefusal(answer) {
  if (!answer) return false
  const opening = answer.replace(/[*_#>]/g, '').trim().split(/(?<=[.!?])\s/)[0]
  return REFUSAL.test(opening)
}
