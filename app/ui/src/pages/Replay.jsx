import { useStore } from '../store.jsx'
import Explorer from '../components/Explorer.jsx'
import Answers from '../components/Answers.jsx'

export default function Replay() {
  const { study, qi, setQi } = useStore()
  const query = study.queries[qi]
  return (
    <>
      <Explorer query={query} queries={study.queries} qi={qi} onPick={setQi} />
      <Answers query={query} queries={study.queries} onPick={setQi} />
    </>
  )
}
