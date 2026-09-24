import apiClient from './axios';

/**
 * Get company forecasting models registry for the currently active dataset.
 */
export const getCurrentModelsApi = async () => {
  const response = await apiClient.get('/api/models/current');
  return response.data;
};

/**
 * Manually trigger asynchronous model benchmarking and training for a dataset.
 */
export const trainDatasetModelsApi = async (datasetId) => {
  const response = await apiClient.post(`/api/datasets/${datasetId}/train`);
  return response.data;
};
