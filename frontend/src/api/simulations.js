import apiClient from './axios';

/**
 * Run a what-if sales simulation scenario.
 * @param {object} simulationRequest - Scenario parameters
 */
export const runSimulationApi = async (simulationRequest) => {
  const response = await apiClient.post('/api/simulations', simulationRequest);
  return response.data;
};

/**
 * Fetch a previously computed simulation result by ID.
 * @param {string} simulationId
 */
export const getSimulationApi = async (simulationId) => {
  const response = await apiClient.get(`/api/simulations/${simulationId}`);
  return response.data;
};
