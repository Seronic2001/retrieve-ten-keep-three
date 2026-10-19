import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { GDPR_ID, StoreProvider, useStore } from './store.jsx'
import Flap from './components/Flap.jsx'
import DocSwitcher from './components/DocSwitcher.jsx'
import Library from './pages/Library.jsx'
import Chat from './pages/Chat.jsx'
import Flow from './pages/Flow.jsx'
import Space from './pages/Space.jsx'
import Study, { StudyOverview } from './pages/Study.jsx'
import Models from './pages/Models.jsx'
import Replay from './pages/Replay.jsx'
import Results from './pages/Results.jsx'
import CostPage from './pages/CostPage.jsx'

function Shell() {
  const { data, error, server, library, libraryReady, activeDoc, study, studyState } = useStore()
  if (error) return <p className="loading">{error}</p>
  if (!data || !libraryReady) return <p className="loading">Loading…</p>

  // Each tab carries a small live readout of what is behind it.
  const count = (s) => study.queries.filter((q) => q.status === s).length
  const studyKick = study
    ? `${activeDoc?.id === GDPR_ID ? 'GDPR · ' : ''}${count('restored')} ✓ ${count('degraded')} ✗`
    : studyState.state === 'building' ? 'building…' : 'build one'
  const pages = [
    ['/', 'Library', `${library.length} PDF${library.length === 1 ? '' : 's'}`],
    ['/chat', 'Ask', <span key="s" className={`kick-dot ${server.state}`}>{server.state === 'live' ? 'live' : server.state === 'offline' ? 'offline' : '…'}</span>],
    ['/flow', 'Chunk flow', activeDoc?.chunks ? `${activeDoc.chunks} → 3` : '… → 3'],
    ['/space', 'Chunk space', activeDoc?.chunks ? `${activeDoc.chunks} points` : 'map'],
    ['/study', 'Case study', studyKick],
  ]

  return (
    <>
      <header className="topbar">
        <div className="wrap">
          <NavLink to="/" className="brand" end aria-label="Library">
            <span className="brand-k">Retrieve</span><Flap text="10" /><span className="brand-k">keep</span><Flap text="3" />
          </NavLink>
          <DocSwitcher />
          <nav className="tabsnav" aria-label="Pages">
            {pages.map(([to, label, kick]) => (
              <NavLink key={to} to={to} end={to === '/'} className="itab">
                <small>{kick}</small>
                {label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main>
        <Routes>
          <Route path="/" element={<Library />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/flow" element={<Flow />} />
          <Route path="/space" element={<Space />} />
          <Route path="/study" element={<Study />}>
            <Route index element={<StudyOverview />} />
            <Route path="models" element={<Models />} />
            <Route path="replay" element={<Replay />} />
            <Route path="results" element={<Results />} />
            <Route path="cost" element={<CostPage />} />
          </Route>
          {/* old single-study paths */}
          <Route path="/models" element={<Navigate to="/study/models" replace />} />
          <Route path="/replay" element={<Navigate to="/study/replay" replace />} />
          <Route path="/results" element={<Navigate to="/study/results" replace />} />
          <Route path="/cost" element={<Navigate to="/study/cost" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      <footer>
        <div className="wrap">
          <span>Retrieve 10 · keep 3 — bi-encoder retrieval vs. cross-encoder re-ranking over any PDF</span>
          <span className="mono">{data.models.bi} · {data.models.cross}</span>
        </div>
      </footer>
    </>
  )
}

export default function App() {
  return (
    <StoreProvider>
      <Shell />
    </StoreProvider>
  )
}
