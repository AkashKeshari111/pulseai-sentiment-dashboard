/**
 * Catches render errors from one page instead of blanking the whole app.
 *
 * React unmounts the entire tree when a render throws, so without a boundary a
 * single bad property reference on one page takes the navigation down with it
 * and leaves a white screen with no way back. This keeps the shell alive, shows
 * what actually broke, and lets the user carry on elsewhere.
 */

import { Component } from 'react'
import { AlertTriangle, RotateCcw } from 'lucide-react'
import { Button, Card, CardBody } from './ui'

export class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    // Kept on the console so the stack is still available to a developer.
    console.error('Page crashed:', error, info?.componentStack)
  }

  componentDidUpdate(previousProps) {
    // A route change should clear a previous page's error, otherwise the
    // boundary keeps showing it after the user navigates away.
    if (previousProps.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null })
    }
  }

  render() {
    const { error } = this.state
    if (!error) return this.props.children

    return (
      <Card className="mx-auto max-w-2xl">
        <CardBody className="!pt-6 text-center">
          <AlertTriangle
            className="mx-auto h-7 w-7"
            style={{ color: 'var(--status-serious)' }}
            aria-hidden
          />
          <h2 className="mt-3 text-[15px] font-semibold">This page failed to render</h2>
          <p className="mt-2 text-[13px] leading-relaxed text-[var(--text-secondary)]">
            The rest of the dashboard is still working — use the navigation to carry on.
            If this page depends on training results, the model may still be training.
          </p>
          <pre className="mt-4 overflow-x-auto rounded-lg bg-[var(--surface-2)] p-3 text-left text-[11.5px] text-[var(--text-muted)]">
            {String(error?.message ?? error)}
          </pre>
          <Button
            variant="subtle"
            className="mt-4"
            onClick={() => this.setState({ error: null })}
          >
            <RotateCcw className="h-3.5 w-3.5" aria-hidden />
            Try again
          </Button>
        </CardBody>
      </Card>
    )
  }
}
