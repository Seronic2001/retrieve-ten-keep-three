import { useEffect, useId, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useStore } from '../store.jsx'

/** The document Ask and Chunk flow work on. A themed menu instead of a native <select>, whose popup
 *  the browser draws in its own colours. */
export default function DocSwitcher() {
  const { library, activeDoc, setActiveId } = useStore()
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)
  const [focus, setFocus] = useState(0)
  const root = useRef(null)
  const listId = useId()

  // Ready documents can be picked; the last row adds a new PDF.
  const items = [
    ...library.map((d) => ({ kind: 'doc', doc: d, disabled: d.status !== 'ready' })),
    { kind: 'add' },
  ]
  const choose = (item) => {
    if (!item || item.disabled) return
    setOpen(false)
    if (item.kind === 'add') navigate('/')
    else setActiveId(item.doc.id)
  }

  useEffect(() => {
    if (!open) return
    const away = (e) => { if (!root.current?.contains(e.target)) setOpen(false) }
    document.addEventListener('pointerdown', away)
    return () => document.removeEventListener('pointerdown', away)
  }, [open])

  const openMenu = () => {
    setFocus(Math.max(0, items.findIndex((i) => i.doc?.id === activeDoc?.id)))
    setOpen(true)
  }
  const step = (dir) => {
    let i = focus
    do { i = (i + dir + items.length) % items.length } while (items[i].disabled && i !== focus)
    setFocus(i)
  }
  const onKey = (e) => {
    if (!open) {
      if (['ArrowDown', 'ArrowUp', 'Enter', ' '].includes(e.key)) { e.preventDefault(); openMenu() }
      return
    }
    if (e.key === 'Escape') { e.preventDefault(); setOpen(false) }
    else if (e.key === 'ArrowDown') { e.preventDefault(); step(1) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); step(-1) }
    else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(items[focus]) }
    else if (e.key === 'Tab') setOpen(false)
  }

  if (!activeDoc) return null
  return (
    <div className={`doc-switch ${open ? 'open' : ''}`} ref={root} onKeyDown={onKey}>
      <button
        type="button"
        className="doc-switch-btn"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listId}
        onClick={() => (open ? setOpen(false) : openMenu())}
        title="The document Ask and Chunk flow work on"
      >
        <span className="doc-switch-k">Source</span>
        <span className="doc-switch-title">{activeDoc.title}</span>
        {activeDoc.chunks != null && <span className="doc-switch-meta">{activeDoc.chunks} ch</span>}
        <svg className="doc-switch-caret" width="10" height="10" viewBox="0 0 10 10" aria-hidden="true">
          <path d="M1.5 3.5 5 7l3.5-3.5" fill="none" stroke="currentColor" strokeWidth="1.6" />
        </svg>
      </button>

      {open && (
        <div className="doc-menu" role="listbox" id={listId} aria-label="Documents" aria-activedescendant={`${listId}-${focus}`}>
          <p className="doc-menu-k">Documents · {library.length}</p>
          {items.map((item, i) => {
            if (item.kind === 'add') {
              return (
                <div key="add" id={`${listId}-${i}`} role="option" aria-selected={false}
                  className={`doc-opt add ${focus === i ? 'focus' : ''}`}
                  onMouseEnter={() => setFocus(i)} onClick={() => choose(item)}>
                  <span className="doc-opt-plus" aria-hidden="true">+</span>
                  <span className="doc-opt-main">
                    <span className="doc-opt-title">Add a PDF</span>
                    <span className="doc-opt-meta">Upload and index a new document</span>
                  </span>
                </div>
              )
            }
            const d = item.doc
            const active = d.id === activeDoc.id
            const busy = d.status === 'queued' || d.status === 'indexing'
            return (
              <div key={d.id} id={`${listId}-${i}`} role="option" aria-selected={active} aria-disabled={item.disabled}
                className={`doc-opt ${active ? 'active' : ''} ${focus === i ? 'focus' : ''} ${item.disabled ? 'disabled' : ''}`}
                onMouseEnter={() => !item.disabled && setFocus(i)} onClick={() => choose(item)}>
                <span className={`doc-opt-dot ${d.status}`} aria-hidden="true" />
                <span className="doc-opt-main">
                  <span className="doc-opt-title">{d.title}</span>
                  <span className="doc-opt-meta">
                    {busy
                      ? `indexing · ${Math.round((d.progress ?? 0) * 100)}%`
                      : d.status === 'error'
                        ? 'could not be indexed'
                        : `${d.pages ?? '–'} pages · ${d.chunks ?? '–'} chunks`}
                  </span>
                </span>
                {active && (
                  <svg className="doc-opt-check" width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
                    <path d="M2.5 7.5 5.5 10.5 11.5 3.5" fill="none" stroke="currentColor" strokeWidth="2" />
                  </svg>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
