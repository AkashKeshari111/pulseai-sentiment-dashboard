/**
 * Shared UI primitives.
 *
 * Everything here is styled from the CSS custom properties defined in
 * index.css, so the whole app themes from one file and no component hard-codes
 * a colour.
 */

import clsx from 'clsx'
import { AlertTriangle, Inbox, Loader2 } from 'lucide-react'

/* -------------------------------------------------------------------------- */
/* Surfaces                                                                    */
/* -------------------------------------------------------------------------- */

export function Card({ className, children, ...props }) {
  return (
    <section
      className={clsx(
        // Column flex so a CardBody can claim the leftover height. Grid rows
        // stretch their cards to match the tallest sibling, and without this
        // the shorter card's content sits against the top with dead space
        // below it.
        'flex flex-col rounded-xl border bg-[var(--surface-1)] border-[var(--border)]',
        className,
      )}
      {...props}
    >
      {children}
    </section>
  )
}

export function CardHeader({ title, subtitle, actions, className }) {
  return (
    <header
      className={clsx(
        'flex flex-wrap items-start justify-between gap-3 px-5 pt-4 pb-3',
        className,
      )}
    >
      <div className="min-w-0">
        <h2 className="text-[13px] font-semibold tracking-wide uppercase text-[var(--text-secondary)]">
          {title}
        </h2>
        {subtitle && (
          <p className="mt-1 text-[12px] leading-snug text-[var(--text-muted)]">{subtitle}</p>
        )}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </header>
  )
}

export function CardBody({ className, children }) {
  return <div className={clsx('flex-1 px-5 pb-5', className)}>{children}</div>
}

/* -------------------------------------------------------------------------- */
/* Controls                                                                    */
/* -------------------------------------------------------------------------- */

const buttonVariants = {
  primary:
    'bg-[var(--accent)] text-[var(--accent-contrast)] border-transparent hover:opacity-90',
  ghost:
    'bg-transparent text-[var(--text-secondary)] border-[var(--border)] hover:bg-[var(--surface-2)]',
  subtle:
    'bg-[var(--surface-2)] text-[var(--text-primary)] border-transparent hover:border-[var(--border-strong)]',
  danger:
    'bg-transparent text-[var(--status-critical)] border-[var(--border)] hover:bg-[var(--sentiment-negative-soft)]',
}

export function Button({ variant = 'ghost', className, disabled, children, ...props }) {
  return (
    <button
      type="button"
      disabled={disabled}
      className={clsx(
        'inline-flex items-center justify-center gap-1.5 rounded-lg border px-3 py-1.5',
        'text-[13px] font-medium transition-colors',
        'disabled:cursor-not-allowed disabled:opacity-45',
        buttonVariants[variant],
        className,
      )}
      {...props}
    >
      {children}
    </button>
  )
}

export function Select({ label, value, onChange, options, className, id }) {
  const selectId = id ?? `select-${label?.replace(/\s+/g, '-').toLowerCase()}`
  return (
    <label htmlFor={selectId} className={clsx('flex flex-col gap-1', className)}>
      {label && (
        <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">
          {label}
        </span>
      )}
      <select
        id={selectId}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className={clsx(
          'rounded-lg border border-[var(--border)] bg-[var(--surface-1)] px-2.5 py-1.5',
          'text-[13px] text-[var(--text-primary)] transition-colors',
          'hover:border-[var(--border-strong)]',
        )}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  )
}

export function TextInput({ label, className, id, ...props }) {
  const inputId = id ?? `input-${label?.replace(/\s+/g, '-').toLowerCase()}`
  return (
    <label htmlFor={inputId} className={clsx('flex flex-col gap-1', className)}>
      {label && (
        <span className="text-[11px] font-medium uppercase tracking-wide text-[var(--text-muted)]">
          {label}
        </span>
      )}
      <input
        id={inputId}
        className={clsx(
          'rounded-lg border border-[var(--border)] bg-[var(--surface-1)] px-2.5 py-1.5',
          'text-[13px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)]',
          'transition-colors hover:border-[var(--border-strong)]',
        )}
        {...props}
      />
    </label>
  )
}

/* -------------------------------------------------------------------------- */
/* Status blocks                                                               */
/* -------------------------------------------------------------------------- */

export function Spinner({ className }) {
  return <Loader2 className={clsx('h-4 w-4 animate-spin', className)} aria-hidden />
}

export function Skeleton({ className }) {
  return <div className={clsx('skeleton rounded-lg', className)} aria-hidden />
}

export function LoadingBlock({ height = 'h-64', label = 'Loading' }) {
  return (
    <div className={clsx('flex w-full items-center justify-center', height)} role="status">
      <span className="flex items-center gap-2 text-[13px] text-[var(--text-muted)]">
        <Spinner />
        {label}…
      </span>
    </div>
  )
}

export function EmptyState({ title, hint, icon: Icon = Inbox, action }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-12 text-center">
      <Icon className="h-6 w-6 text-[var(--text-muted)]" aria-hidden />
      <p className="text-[14px] font-medium text-[var(--text-secondary)]">{title}</p>
      {hint && <p className="max-w-md text-[12.5px] leading-relaxed text-[var(--text-muted)]">{hint}</p>}
      {action}
    </div>
  )
}

export function ErrorState({ error, onRetry }) {
  const message = error?.message ?? 'Something went wrong.'
  const isOffline = error?.status === 503

  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-10 text-center">
      <AlertTriangle className="h-6 w-6 text-[var(--status-serious)]" aria-hidden />
      <p className="text-[14px] font-medium text-[var(--text-secondary)]">
        {isOffline ? 'Database not connected' : 'Request failed'}
      </p>
      <p className="max-w-lg text-[12.5px] leading-relaxed text-[var(--text-muted)]">{message}</p>
      {onRetry && (
        <Button variant="subtle" onClick={onRetry} className="mt-2">
          Try again
        </Button>
      )}
    </div>
  )
}

/**
 * Wraps a data-backed panel so loading, error and empty are handled the same
 * way everywhere instead of being re-implemented per chart.
 */
export function QueryBoundary({ query, isEmpty, empty, height, children }) {
  if (query.isLoading) return <LoadingBlock height={height} />
  if (query.isError) return <ErrorState error={query.error} onRetry={query.refetch} />
  if (isEmpty?.(query.data)) {
    return (
      empty ?? (
        <EmptyState
          title="No feedback matches these filters"
          hint="Try widening the date range, or seed the database with `python -m api.seed`."
        />
      )
    )
  }
  return children(query.data)
}
