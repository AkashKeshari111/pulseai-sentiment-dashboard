/**
 * React Query hooks.
 *
 * Every analytics hook keys its cache on the active filter object, so changing
 * a filter refetches exactly the affected queries and returns instantly when
 * the user navigates back to a combination already in cache.
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from '../lib/api'
import { useFilters } from '../lib/filters'

const FRESH = 30_000 // analytics tolerate half a minute of staleness

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: api.health,
    refetchInterval: 30_000,
    retry: 1,
  })
}

export function useModelMetrics() {
  return useQuery({
    queryKey: ['model-metrics'],
    queryFn: api.modelMetrics,
    // Offline evaluation results only change when training is re-run.
    staleTime: 5 * 60_000,
  })
}

export function useFilterOptions() {
  return useQuery({
    queryKey: ['filter-options'],
    queryFn: api.filterOptions,
    staleTime: 5 * 60_000,
    retry: 1,
  })
}

export function useSummary() {
  const { query } = useFilters()
  return useQuery({
    queryKey: ['summary', query],
    queryFn: () => api.summary(query),
    staleTime: FRESH,
  })
}

export function useTrends(granularity = 'day') {
  const { query } = useFilters()
  return useQuery({
    queryKey: ['trends', query, granularity],
    queryFn: () => api.trends({ ...query, granularity }),
    staleTime: FRESH,
  })
}

export function useSources() {
  const { query } = useFilters()
  return useQuery({
    queryKey: ['sources', query],
    queryFn: () => api.sources(query),
    staleTime: FRESH,
  })
}

export function useProducts() {
  const { query } = useFilters()
  return useQuery({
    queryKey: ['products', query],
    queryFn: () => api.products(query),
    staleTime: FRESH,
  })
}

export function useIssues(limit = 8) {
  const { query } = useFilters()
  return useQuery({
    queryKey: ['issues', query, limit],
    queryFn: () => api.issues({ ...query, limit }),
    staleTime: FRESH,
  })
}

export function useKeywords(sentiment = 'all', limit = 45) {
  const { query } = useFilters()
  return useQuery({
    queryKey: ['keywords', query, sentiment, limit],
    queryFn: () => api.keywords({ ...query, sentiment, limit }),
    staleTime: FRESH,
  })
}

export function useFeedbackList({ page, pageSize, sort }) {
  const { query } = useFilters()
  return useQuery({
    queryKey: ['feedback', query, page, pageSize, sort],
    queryFn: () => api.listFeedback({ ...query, page, page_size: pageSize, sort }),
    // Keeping the previous page visible while the next loads avoids the table
    // collapsing to a spinner on every pagination click.
    placeholderData: (previous) => previous,
    staleTime: 10_000,
  })
}

/** Invalidate every data-backed query - used after an ingest or a delete. */
export function useRefreshAnalytics() {
  const queryClient = useQueryClient()
  return () =>
    queryClient.invalidateQueries({
      predicate: (query) =>
        ['summary', 'trends', 'sources', 'products', 'issues', 'keywords', 'feedback', 'health']
          .includes(query.queryKey[0]),
    })
}

export function useCreateFeedback() {
  const refresh = useRefreshAnalytics()
  return useMutation({
    mutationFn: (item) => api.createFeedback(item),
    onSuccess: refresh,
  })
}

export function useDeleteFeedback() {
  const refresh = useRefreshAnalytics()
  return useMutation({
    mutationFn: (id) => api.deleteFeedback(id),
    onSuccess: refresh,
  })
}

export function useUploadCsv() {
  const refresh = useRefreshAnalytics()
  return useMutation({
    mutationFn: ({ file, source }) => api.uploadCsv(file, source),
    onSuccess: refresh,
  })
}

export function usePredict() {
  return useMutation({
    mutationFn: ({ text, explain }) => api.predict(text, explain),
  })
}
