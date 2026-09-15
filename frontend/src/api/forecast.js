import apiClient from './axios';

/**
 * Generate demand forecast for 7, 30, or 90 days.
 */
export const getForecastApi = async (horizon = 7) => {
  const response = await apiClient.post('/api/forecast', { horizon });
  return response.data;
};

/**
 * Fetch historical sales, demand, and profit data.
 */
export const getForecastHistoryApi = async (limit = 365) => {
  const response = await apiClient.get('/api/forecast/history', {
    params: { limit },
  });
  return response.data;
};

/**
 * Fetch combined executive dashboard payload: KPIs, history, forecast, model name.
 */
export const getDashboardApi = async (horizon = 7, historyLimit = 365) => {
  const response = await apiClient.get('/api/forecast/dashboard', {
    params: {
      horizon,
      history_limit: historyLimit,
    },
  });
  return response.data;
};
