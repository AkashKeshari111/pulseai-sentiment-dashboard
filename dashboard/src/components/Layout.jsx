/** Application shell: sidebar navigation, top bar, service status. */

import { useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import clsx from 'clsx'
import {
  Activity,
  BarChart3,
  Brain,
  Database,
  Lightbulb,
  Menu,
  Moon,
  Sun,
  Table2,
  Wand2,
  X,
} from 'lucide-react'
import { ErrorBoundary } from './ErrorBoundary'
import { useTheme } from '../lib/theme'
import { useHealth } from '../hooks/useAnalytics'

const NAV = [
  { to: '/', label: 'Overview', icon: BarChart3, end: true, hint: 'KPIs, trend and live feed' },
  { to: '/explorer', label: 'Explorer', icon: Table2, hint: 'Browse and filter every record' },
  { to: '/insights', label: 'Insights', icon: Lightbulb, hint: 'Issue drivers and language' },
  { to: '/analyze', label: 'Analyze', icon: Wand2, hint: 'Classify new feedback live' },
  { to: '/model', label: 'Model card', icon: Brain, hint: 'Evaluation and training results' },
]

function StatusPill() {
  const { data, isLoading, isError } = useHealth()

  const connected = data?.database?.connected
  const state = isLoading
    ? { text: 'Checking…', color: 'var(--text-muted)' }
    : isError
      ? { text: 'API offline', color: 'var(--status-critical)' }
      : connected
        ? { text: 'Connected', color: 'var(--status-good)' }
        : { text: 'No database', color: 'var(--status-warning)' }

  const title = isError
    ? 'The dashboard cannot reach the FastAPI service. Start it with: uvicorn api.main:app --reload'
    : connected
      ? `MongoDB Atlas · ${data?.database?.documents ?? 0} documents`
      : (data?.database?.error ?? 'MongoDB is not configured')

  return (
    <span
      title={title}
      className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--border)] px-2.5 py-1.5 text-[12px] text-[var(--text-secondary)]"
    >
      <Database className="h-3.5 w-3.5" aria-hidden style={{ color: state.color }} />
      <span className="hidden sm:inline">{state.text}</span>
    </span>
  )
}

function ModelPill() {
  const { data } = useHealth()
  const model = data?.model
  if (!model) return null

  const isFineTuned = model.source === 'local' && model.backend === 'transformer'
  return (
    <span
      title={`${model.model} · backend=${model.backend} · device=${model.device}`}
      className="hidden items-center gap-1.5 rounded-lg border border-[var(--border)] px-2.5 py-1.5 text-[12px] text-[var(--text-secondary)] md:inline-flex"
    >
      <Activity
        className="h-3.5 w-3.5"
        aria-hidden
        style={{ color: isFineTuned ? 'var(--status-good)' : 'var(--status-warning)' }}
      />
      <span className="max-w-[190px] truncate">{model.model}</span>
    </span>
  )
}

function SidebarContent({ onNavigate }) {
  return (
    <>
      <div className="flex items-center gap-2.5 px-5 py-5">
        <span
          className="grid h-8 w-8 place-items-center rounded-lg text-[15px] font-bold"
          style={{ background: 'var(--accent)', color: 'var(--accent-contrast)' }}
          aria-hidden
        >
          P
        </span>
        <span className="min-w-0">
          <span className="block text-[14px] font-semibold leading-tight">PulseAI</span>
          <span className="block text-[11px] text-[var(--text-muted)]">
            Sentiment Intelligence
          </span>
        </span>
      </div>

      <nav className="flex flex-col gap-0.5 px-3" aria-label="Main">
        {NAV.map(({ to, label, icon: Icon, end, hint }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            onClick={onNavigate}
            title={hint}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13.5px] transition-colors',
                isActive
                  ? 'bg-[var(--surface-2)] font-medium text-[var(--text-primary)]'
                  : 'text-[var(--text-secondary)] hover:bg-[var(--surface-2)]',
              )
            }
          >
            <Icon className="h-4 w-4 shrink-0" aria-hidden />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto px-5 py-5 text-[11px] leading-relaxed text-[var(--text-muted)]">
        <p>Fine-tuned DistilBERT · FastAPI · MongoDB Atlas</p>
        <a
          href="/docs"
          target="_blank"
          rel="noreferrer"
          className="mt-1 inline-block underline decoration-dotted underline-offset-2 hover:text-[var(--text-secondary)]"
        >
          API documentation
        </a>
      </div>
    </>
  )
}

export function Layout() {
  const { theme, toggle } = useTheme()
  const [mobileOpen, setMobileOpen] = useState(false)
  // Navigating away clears a crashed page's error state.
  const { pathname } = useLocation()

  return (
    <div className="flex min-h-screen">
      {/* Desktop sidebar */}
      <aside className="hidden w-60 shrink-0 flex-col border-r border-[var(--border)] bg-[var(--surface-1)] lg:flex">
        <SidebarContent />
      </aside>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div
            className="absolute inset-0"
            style={{ background: 'var(--overlay)' }}
            onClick={() => setMobileOpen(false)}
            aria-hidden
          />
          <aside className="absolute inset-y-0 left-0 flex w-64 flex-col border-r border-[var(--border)] bg-[var(--surface-1)]">
            <SidebarContent onNavigate={() => setMobileOpen(false)} />
          </aside>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex items-center gap-3 border-b border-[var(--border)] bg-[var(--page)]/85 px-4 py-3 backdrop-blur">
          <button
            type="button"
            onClick={() => setMobileOpen((open) => !open)}
            className="rounded-lg border border-[var(--border)] p-1.5 text-[var(--text-secondary)] lg:hidden"
            aria-label={mobileOpen ? 'Close navigation' : 'Open navigation'}
          >
            {mobileOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
          </button>

          <div className="mr-auto min-w-0">
            <h1 className="truncate text-[15px] font-semibold">
              Customer Sentiment Intelligence
            </h1>
            <p className="hidden text-[12px] text-[var(--text-muted)] sm:block">
              Real-time classification and trend analysis of customer feedback
            </p>
          </div>

          <ModelPill />
          <StatusPill />
          <button
            type="button"
            onClick={toggle}
            className="rounded-lg border border-[var(--border)] p-1.5 text-[var(--text-secondary)] hover:bg-[var(--surface-2)]"
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
          >
            {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>
        </header>

        <main className="min-w-0 flex-1 px-4 py-5 sm:px-6">
          <ErrorBoundary resetKey={pathname}>
            <Outlet />
          </ErrorBoundary>
        </main>
      </div>
    </div>
  )
}
