import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { getModels } from '@/services/api'

const links = [
  { to: '/training-data', label: 'Training Data' },
  { to: '/training', label: 'Training' },
  { to: '/testing', label: 'Testing' },
  { to: '/models', label: 'Models' },
]

export default function Navbar() {
  const [activeModel, setActiveModel] = useState(null)

  useEffect(() => {
    getModels()
      .then(({ data }) => {
        const active = data.find((m) => m.is_active)
        if (active) setActiveModel(active)
      })
      .catch(() => {})
  }, [])

  return (
    <nav className="border-b bg-background px-6 py-3 flex items-center gap-6">
      <span className="font-semibold text-lg tracking-tight shrink-0">NIC OCR</span>

      <div className="flex items-center gap-1 flex-1">
        {links.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              [
                'px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:text-foreground hover:bg-accent',
              ].join(' ')
            }
          >
            {label}
          </NavLink>
        ))}
      </div>

      {activeModel && (
        <span className="shrink-0 text-xs bg-green-100 text-green-800 border border-green-300 rounded-full px-2.5 py-0.5 font-medium">
          Active: {activeModel.id}
        </span>
      )}
    </nav>
  )
}
