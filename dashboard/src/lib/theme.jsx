/**
 * Theme state and chart colour resolution.
 *
 * Charts are rendered by Recharts into SVG, and Recharts needs literal colour
 * values rather than `var(--x)` for some props (gradient stops, cell fills).
 * `useChartColors` reads the resolved custom properties off the document once
 * per theme change and hands components plain hex, so the palette still lives
 * in exactly one place - index.css - and never drifts between CSS and JS.
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'

const STORAGE_KEY = 'pulseai-theme'
const ThemeContext = createContext(null)

function readStoredTheme() {
  if (typeof window === 'undefined') return 'dark'
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY)
    if (stored === 'light' || stored === 'dark') return stored
  } catch {
    /* private mode / blocked storage - fall through to the OS preference */
  }
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(readStoredTheme)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    try {
      window.localStorage.setItem(STORAGE_KEY, theme)
    } catch {
      /* nothing to do - the attribute above already applied the theme */
    }
  }, [theme])

  const toggle = useCallback(
    () => setTheme((current) => (current === 'dark' ? 'light' : 'dark')),
    [],
  )

  const value = useMemo(() => ({ theme, setTheme, toggle }), [theme, toggle])
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>
}

export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) throw new Error('useTheme must be used inside <ThemeProvider>')
  return context
}

const TOKENS = {
  negative: '--sentiment-negative',
  neutral: '--sentiment-neutral',
  positive: '--sentiment-positive',
  series1: '--series-1',
  series2: '--series-2',
  series3: '--series-3',
  seq100: '--seq-100',
  seq250: '--seq-250',
  seq400: '--seq-400',
  seq550: '--seq-550',
  seq700: '--seq-700',
  good: '--status-good',
  warning: '--status-warning',
  serious: '--status-serious',
  critical: '--status-critical',
  grid: '--gridline',
  baseline: '--baseline',
  surface: '--surface-1',
  surface2: '--surface-2',
  text: '--text-primary',
  textSecondary: '--text-secondary',
  muted: '--text-muted',
  border: '--border',
  accent: '--accent',
}

export function useChartColors() {
  const { theme } = useTheme()

  return useMemo(() => {
    if (typeof window === 'undefined') return {}
    const styles = getComputedStyle(document.documentElement)
    const resolved = {}
    for (const [name, token] of Object.entries(TOKENS)) {
      resolved[name] = styles.getPropertyValue(token).trim()
    }
    // Ordered ramp for magnitude encodings (confusion matrix cells).
    resolved.sequential = [
      resolved.seq100,
      resolved.seq250,
      resolved.seq400,
      resolved.seq550,
      resolved.seq700,
    ]
    resolved.sentiment = {
      negative: resolved.negative,
      neutral: resolved.neutral,
      positive: resolved.positive,
    }
    // `theme` is the dependency that makes this recompute on toggle.
    resolved.theme = theme
    return resolved
  }, [theme])
}
