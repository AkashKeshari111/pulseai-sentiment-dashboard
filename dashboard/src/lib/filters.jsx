/**
 * Global filter state.
 *
 * One source of truth for the date window, channel, category, sentiment and
 * search term. Every KPI, chart and table on a page reads the same object, so
 * the numbers can never disagree with each other - which is the failure mode
 * that makes an analytics dashboard untrustworthy.
 *
 * The state is mirrored into the URL query string so a filtered view can be
 * shared, bookmarked, and survives a refresh.
 */

import { createContext, useCallback, useContext, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'

const FilterContext = createContext(null)

export const DATE_RANGES = [
  { value: '7', label: 'Last 7 days' },
  { value: '30', label: 'Last 30 days' },
  { value: '90', label: 'Last 90 days' },
  { value: '365', label: 'Last 12 months' },
  { value: 'all', label: 'All time' },
]

const DEFAULTS = {
  days: '90',
  sentiment: 'all',
  source: 'all',
  category: 'all',
  search: '',
}

export function FilterProvider({ children }) {
  const [searchParams, setSearchParams] = useSearchParams()

  const filters = useMemo(() => {
    const current = { ...DEFAULTS }
    for (const key of Object.keys(DEFAULTS)) {
      const value = searchParams.get(key)
      if (value !== null) current[key] = value
    }
    return current
  }, [searchParams])

  const setFilter = useCallback(
    (key, value) => {
      setSearchParams(
        (previous) => {
          const next = new URLSearchParams(previous)
          if (value === DEFAULTS[key] || value === '' || value === null) {
            next.delete(key)
          } else {
            next.set(key, value)
          }
          // Any filter change invalidates the current page of the table.
          next.delete('page')
          return next
        },
        { replace: true },
      )
    },
    [setSearchParams],
  )

  const reset = useCallback(() => setSearchParams({}, { replace: true }), [setSearchParams])

  /** The subset the API understands, with UI-only sentinels stripped. */
  const query = useMemo(() => {
    const params = {}
    if (filters.days !== 'all') params.days = filters.days
    if (filters.sentiment !== 'all') params.sentiment = filters.sentiment
    if (filters.source !== 'all') params.source = filters.source
    if (filters.category !== 'all') params.category = filters.category
    if (filters.search.trim()) params.search = filters.search.trim()
    return params
  }, [filters])

  const activeCount = useMemo(
    () => Object.keys(DEFAULTS).filter((key) => filters[key] !== DEFAULTS[key]).length,
    [filters],
  )

  const value = useMemo(
    () => ({ filters, setFilter, reset, query, activeCount }),
    [filters, setFilter, reset, query, activeCount],
  )
  return <FilterContext.Provider value={value}>{children}</FilterContext.Provider>
}

export function useFilters() {
  const context = useContext(FilterContext)
  if (!context) throw new Error('useFilters must be used inside <FilterProvider>')
  return context
}
