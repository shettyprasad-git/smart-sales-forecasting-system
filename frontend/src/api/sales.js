import apiClient from './axios';

/**
 * Fetch list of sales records.
 */
export const getSalesApi = async (skip = 0, limit = 100) => {
  const response = await apiClient.get('/api/sales', {
    params: { skip, limit },
  });
  return response.data;
};

/**
 * Get sales record by ID.
 */
export const getSaleByIdApi = async (salesId) => {
  const response = await apiClient.get(`/api/sales/${salesId}`);
  return response.data;
};

/**
 * Create a sales record.
 * NOTE: payload.product_id MUST be the database integer ID of the Product.
 */
export const createSaleApi = async (salesData) => {
  const response = await apiClient.post('/api/sales', salesData);
  return response.data;
};

/**
 * Update a sales record.
 */
export const updateSaleApi = async (salesId, salesData) => {
  const response = await apiClient.put(`/api/sales/${salesId}`, salesData);
  return response.data;
};

/**
 * Delete a sales record.
 */
export const deleteSaleApi = async (salesId) => {
  const response = await apiClient.delete(`/api/sales/${salesId}`);
  return response.data;
};
