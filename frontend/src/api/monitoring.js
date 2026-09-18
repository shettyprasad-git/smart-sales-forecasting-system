import api from './axios';

export const runMonitoring = async (params = {}) => {
  const response = await api.post('/api/monitoring/run', params);
  return response.data;
};

export const getAlerts = async (params = {}) => {
  const response = await api.get('/api/monitoring/alerts', { params });
  return response.data;
};

export const getAlert = async (alertId) => {
  const response = await api.get(`/api/monitoring/alerts/${alertId}`);
  return response.data;
};

export const acknowledgeAlert = async (alertId) => {
  const response = await api.post(`/api/monitoring/alerts/${alertId}/acknowledge`);
  return response.data;
};

export const resolveAlert = async (alertId) => {
  const response = await api.post(`/api/monitoring/alerts/${alertId}/resolve`);
  return response.data;
};

export const dismissAlert = async (alertId) => {
  const response = await api.post(`/api/monitoring/alerts/${alertId}/dismiss`);
  return response.data;
};

export const getMonitoringSummary = async () => {
  const response = await api.get('/api/monitoring/summary');
  return response.data;
};

export const getMonitoringRuns = async (params = {}) => {
  const response = await api.get('/api/monitoring/runs', { params });
  return response.data;
};
