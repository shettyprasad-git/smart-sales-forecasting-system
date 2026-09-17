import apiClient from './axios';

/**
 * Fetch AI-grounded commercial reasoning for an anomaly.
 * @param {string} anomalyId - Unique anomaly identifier
 * @param {object} params - Optional parameters (e.g. { refresh: true })
 */
export const getAIReasoningApi = async (anomalyId, params = {}) => {
  const response = await apiClient.get(`/api/ai-reasoning/${anomalyId}`, {
    params,
  });
  return response.data;
};
