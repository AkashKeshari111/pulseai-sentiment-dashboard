/**
 * Subscribes to the API's Server-Sent Events feed.
 *
 * `EventSource` reconnects on its own, so there is no retry logic here - the
 * hook only has to keep a bounded buffer and clean up on unmount. The server
 * replays the last few items on connect, which is why a freshly mounted feed
 * is populated immediately instead of sitting empty until someone submits.
 */

import { useEffect, useRef, useState } from 'react'
import { streamUrl } from '../lib/api'

const MAX_ITEMS = 30

export function useLiveFeed({ enabled = true, replay = 6 } = {}) {
  const [items, setItems] = useState([])
  const [status, setStatus] = useState('connecting')
  const [transport, setTransport] = useState(null)
  const sourceRef = useRef(null)

  useEffect(() => {
    if (!enabled) {
      setStatus('paused')
      return undefined
    }

    const source = new EventSource(streamUrl(replay))
    sourceRef.current = source

    const push = (event, isSeed) => {
      try {
        const payload = JSON.parse(event.data)
        setItems((current) => {
          if (current.some((item) => item.id === payload.id)) return current
          const next = isSeed ? [...current, payload] : [payload, ...current]
          return next.slice(0, MAX_ITEMS)
        })
      } catch {
        /* a malformed frame must not tear down the stream */
      }
    }

    const onFeedback = (event) => push(event, false)
    const onSeed = (event) => push(event, true)
    const onReady = (event) => {
      setStatus('live')
      try {
        setTransport(JSON.parse(event.data).transport)
      } catch {
        setTransport(null)
      }
    }

    source.addEventListener('feedback', onFeedback)
    source.addEventListener('seed', onSeed)
    source.addEventListener('ready', onReady)
    source.onopen = () => setStatus('live')
    source.onerror = () => setStatus('reconnecting')

    return () => {
      source.removeEventListener('feedback', onFeedback)
      source.removeEventListener('seed', onSeed)
      source.removeEventListener('ready', onReady)
      source.close()
      sourceRef.current = null
    }
  }, [enabled, replay])

  return { items, status, transport, clear: () => setItems([]) }
}
