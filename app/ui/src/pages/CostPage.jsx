import { useStore } from '../store.jsx'
import Cost from '../components/Cost.jsx'

export default function CostPage() {
  const { study } = useStore()
  return <Cost data={study} />
}
