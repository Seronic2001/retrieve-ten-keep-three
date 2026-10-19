import { useNavigate } from 'react-router-dom'
import { useStore } from '../store.jsx'
import Scoreboard from '../components/Scoreboard.jsx'
import Hunt from '../components/Hunt.jsx'

export default function Results() {
  const { study, qi, setQi } = useStore()
  const navigate = useNavigate()
  return (
    <>
      <Scoreboard data={study} qi={qi} onPick={(i) => { setQi(i); navigate('/study/replay') }} />
      <Hunt />
    </>
  )
}
