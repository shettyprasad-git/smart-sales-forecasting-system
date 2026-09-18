import apiClient from './axios';

/**
 * Submit a recommendation for human governance review.
 * @param {object} decisionData
 */
export const createDecisionApi = async (decisionData) => {
  const response = await apiClient.post('/api/decisions', decisionData);
  return response.data;
};

/**
 * Retrieve a decision record by ID.
 * @param {string} decisionId
 */
export const getDecisionApi = async (decisionId) => {
  const response = await apiClient.get(`/api/decisions/${decisionId}`);
  return response.data;
};

/**
 * List decision records with optional filters.
 * @param {object} params - { status, anomaly_id, recommendation_type, skip, limit }
 */
export const listDecisionsApi = async (params = {}) => {
  const response = await apiClient.get('/api/decisions', { params });
  return response.data;
};

/**
 * Approve a recommendation for human consideration.
 * @param {string} decisionId
 * @param {object} data - { decision_note, modified_action }
 */
export const approveDecisionApi = async (decisionId, data = {}) => {
  const response = await apiClient.post(`/api/decisions/${decisionId}/approve`, data);
  return response.data;
};

/**
 * Reject a recommendation.
 * @param {string} decisionId
 * @param {object} data - { decision_note }
 */
export const rejectDecisionApi = async (decisionId, data) => {
  const response = await apiClient.post(`/api/decisions/${decisionId}/reject`, data);
  return response.data;
};

/**
 * Request changes or revisions on a recommendation.
 * @param {string} decisionId
 * @param {object} data - { decision_note, modified_action }
 */
export const requestChangesApi = async (decisionId, data) => {
  const response = await apiClient.post(`/api/decisions/${decisionId}/request-changes`, data);
  return response.data;
};

/**
 * Resubmit a decision back to pending review after revision.
 * @param {string} decisionId
 * @param {object} data - { modified_action, decision_note }
 */
export const resubmitDecisionApi = async (decisionId, data = {}) => {
  const response = await apiClient.post(`/api/decisions/${decisionId}/resubmit`, data);
  return response.data;
};
