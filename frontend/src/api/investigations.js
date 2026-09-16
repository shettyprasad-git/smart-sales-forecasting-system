import apiClient from './axios';

/**
 * Fetch root-cause attribution, impact quantification, and evidence for an anomaly.
 */
export const getInvestigationApi = async (anomalyId, params = {}) => {
  const response = await apiClient.get(`/api/investigations/${anomalyId}`, {
    params,
  });
  return response.data;
};
