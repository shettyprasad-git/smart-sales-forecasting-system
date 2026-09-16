import apiClient from './axios';

/**
 * Fetch anomalies with multi-dimensional filtering and pagination.
 */
export const getAnomaliesApi = async (params = {}) => {
  const response = await apiClient.get('/api/anomalies', { params });
  return response.data;
};

/**
 * Fetch aggregated anomaly summary metrics and severity breakdown.
 */
export const getAnomalySummaryApi = async (params = {}) => {
  const response = await apiClient.get('/api/anomalies/summary', { params });
  return response.data;
};

/**
 * Fetch quick list of recent anomalies.
 */
export const getRecentAnomaliesApi = async (limit = 10, params = {}) => {
  const response = await apiClient.get('/api/anomalies/recent', {
    params: { limit, ...params },
  });
  return response.data;
};

/**
 * Fetch product-specific anomalies.
 */
export const getProductAnomaliesApi = async (productId, params = {}) => {
  const response = await apiClient.get(`/api/anomalies/products/${productId}`, {
    params,
  });
  return response.data;
};
