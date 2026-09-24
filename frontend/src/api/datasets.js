import apiClient from './axios';

/**
 * Upload a sales CSV dataset.
 * Expects a FormData object containing 'file'.
 */
export const uploadDatasetApi = async (formData, onUploadProgress) => {
  const response = await apiClient.post('/api/datasets/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress,
  });
  return response.data;
};

/**
 * Get active/current dataset summary for the logged-in user.
 */
export const getCurrentDatasetApi = async () => {
  const response = await apiClient.get('/api/datasets/current');
  return response.data;
};

/**
 * Get dataset upload history for the logged-in user.
 */
export const getDatasetHistoryApi = async () => {
  const response = await apiClient.get('/api/datasets/history');
  return response.data;
};

/**
 * Activate an uploaded dataset by ID.
 */
export const activateDatasetApi = async (datasetId) => {
  const response = await apiClient.post(`/api/datasets/${datasetId}/activate`);
  return response.data;
};

/**
 * Delete an uploaded dataset and its associated sales records.
 */
export const deleteDatasetApi = async (datasetId) => {
  const response = await apiClient.delete(`/api/datasets/${datasetId}`);
  return response.data;
};
