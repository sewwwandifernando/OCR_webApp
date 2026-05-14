import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from '@/components/ui/sonner'
import Navbar from '@/components/Navbar'
import TrainingDataPage from '@/pages/TrainingData'
import TrainingPage from '@/pages/Training'
import TestingPage from '@/pages/Testing'
import ModelsPage from '@/pages/Models'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col">
        <Navbar />
        <Routes>
          <Route path="/" element={<Navigate to="/training-data" replace />} />
          <Route path="/training-data" element={<TrainingDataPage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/testing" element={<TestingPage />} />
          <Route path="/models" element={<ModelsPage />} />
        </Routes>
      </div>
      <Toaster richColors />
    </BrowserRouter>
  )
}
