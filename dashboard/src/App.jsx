import { Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { FilterProvider } from './lib/filters'
import { Analyze } from './pages/Analyze'
import { Explorer } from './pages/Explorer'
import { Insights } from './pages/Insights'
import { ModelCardPage } from './pages/ModelCard'
import { Overview } from './pages/Overview'

export default function App() {
  return (
    // FilterProvider sits inside the router because it reads and writes the
    // query string - filters are part of the URL, so a filtered view is
    // shareable and survives a refresh.
    <FilterProvider>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Overview />} />
          <Route path="explorer" element={<Explorer />} />
          <Route path="insights" element={<Insights />} />
          <Route path="analyze" element={<Analyze />} />
          <Route path="model" element={<ModelCardPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </FilterProvider>
  )
}
