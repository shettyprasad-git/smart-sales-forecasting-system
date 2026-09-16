import apiClient from './axios';

/**
 * Fetch deterministic executive explanation briefing for an anomaly.
 */
export const getExplanationApi = async (anomalyId, params = {}) => {
  const response = await apiClient.get(`/api/explanations/${anomalyId}`, {
    params,
  });
  return response.data;
};
