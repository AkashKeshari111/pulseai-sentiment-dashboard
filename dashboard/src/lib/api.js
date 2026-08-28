/**
 * Typed-ish client for the PulseAI API.
 *
 * Requests go to same-origin relative paths. In development Vite proxies them
 * to the FastAPI process; in production the dashboard is served behind the
 * same host. Either way the client never hard-codes a backend URL, so no
 * rebuild is needed to move environments. `VITE_API_BASE` overrides it when
 * the API genuinely lives elsewhere (e.g. a separate Render service).
 */

const BASE = import.meta.env.VITE_API_BASE ?? ''
const API_KEY = import.meta.env.VITE_API_KEY ?? ''

export class ApiError extends Error {
  constructor(message, status, payload) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.payload = payload
  }
}

function buildQuery(params = {}) {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    // `all` is the dashboard's "no filter" sentinel; sending it would make the
    // API filter on the literal string.
    if (value === undefined || value === null || value === '' || value === 'all') continue
    search.append(key, value)
  }
  const qs = search.toString()
  return qs ? `?${qs}` : ''
}

async function request(path, { method = 'GET', body, params, signal, isForm } = {}) {
  const headers = {}
  if (API_KEY) headers['X-API-Key'] = API_KEY
  if (body && !isForm) headers['Content-Type'] = 'application/json'

  const response = await fetch(`${BASE}${path}${buildQuery(params)}`, {
    method,
    headers,
    signal,
    body: isForm ? body : body ? JSON.stringify(body) : undefined,
  })

  const text = await response.text()
  let payload = null
  try {
    payload = text ? JSON.parse(text) : null
  } catch {
    payload = text
  }

  if (!response.ok) {
    const detail =
      (payload && (payload.detail ?? payload.message)) || response.statusText
    throw new ApiError(
      typeof detail === 'string' ? detail : JSON.stringify(detail),
      response.status,
      payload,
    )
  }
  return payload
}

export const api = {
  health: () => request('/health'),
  modelInfo: () => request('/api/model/info'),
  modelMetrics: () => request('/api/model/metrics'),

  predict: (text, explain = false) =>
    request('/api/predict', { method: 'POST', body: { text, explain } }),
  predictBatch: (texts) =>
    request('/api/predict/batch', { method: 'POST', body: { texts } }),

  createFeedback: (item) =>
    request('/api/feedback', { method: 'POST', body: item }),
  listFeedback: (params) => request('/api/feedback', { params }),
  deleteFeedback: (id) => request(`/api/feedback/${id}`, { method: 'DELETE' }),
  uploadCsv: (file, source = 'csv_upload') => {
    const form = new FormData()
    form.append('file', file)
    return request(`/api/feedback/upload?source=${encodeURIComponent(source)}`, {
      method: 'POST',
      body: form,
      isForm: true,
    })
  },

  summary: (params) => request('/api/analytics/summary', { params }),
  trends: (params) => request('/api/analytics/trends', { params }),
  sources: (params) => request('/api/analytics/sources', { params }),
  products: (params) => request('/api/analytics/products', { params }),
  issues: (params) => request('/api/analytics/issues', { params }),
  keywords: (params) => request('/api/analytics/keywords', { params }),
  recent: (params) => request('/api/analytics/recent', { params }),
  filterOptions: () => request('/api/analytics/filters'),
}

/** URL for the SSE live feed (EventSource cannot send custom headers). */
export const streamUrl = (replay = 6) => `${BASE}/api/stream?replay=${replay}`
