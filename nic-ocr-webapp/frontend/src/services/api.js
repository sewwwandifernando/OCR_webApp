import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const getTrainingData = () => api.get('/training-data')

export const uploadTrainingData = (formData, onProgress) =>
  api.post('/training-data/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: onProgress,
  })

export const updateGroundTruth = (id, ground_truth) => {
  const fd = new FormData()
  fd.append('ground_truth', ground_truth)
  return api.put(`/training-data/${id}/ground-truth`, fd)
}

export const deleteTrainingData = (id) => api.delete(`/training-data/${id}`)

export const getPreviewUrl = (id) => `/api/training-data/${id}/preview`

export const startTraining = (iterations) =>
  api.post('/training/start', { iterations })

export const getTrainingRuns = () => api.get('/training/runs')

export const getTrainingStatus = () => api.get('/training/status')

export const getModels = () => api.get('/models')

export const activateModel = (runId) => api.post(`/models/${runId}/activate`)

export const getModelDownloadUrl = (runId) => `/api/models/${runId}/download`

export const runOcr = (formData) =>
  api.post('/testing/ocr', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
