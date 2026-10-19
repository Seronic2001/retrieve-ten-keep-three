import { useStore } from '../store.jsx'
import Encoders from '../components/Encoders.jsx'
import TestSelect from '../components/TestSelect.jsx'

export default function Models() {
  const { study, qi } = useStore()
  return (
    <>
      <div className="wrap page-top"><TestSelect /></div>
      <Encoders query={study.queries[qi]} />
    </>
  )
}
