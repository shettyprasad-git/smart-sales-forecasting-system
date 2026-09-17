import apiClient from './axios';

/**
 * Fetch prescriptive action recommendations for an anomaly.
 * @param {string} anomalyId - Unique anomaly identifier
 * @param {object} params - Optional parameters (e.g. { refresh: true, fallback: true })
 */
export const getRecommendationsApi = async (anomalyId, params = {}) => {
  const response = await apiClient.get(`/api/recommendations/${anomalyId}`, {
    params,
  });
  return response.data;
};
