import apiClient from './axios';

/**
 * Fetch list of products.
 */
export const getProductsApi = async (skip = 0, limit = 100) => {
  const response = await apiClient.get('/api/products', {
    params: { skip, limit },
  });
  return response.data;
};

/**
 * Get product by external product_id (string, e.g. "PROD-001").
 */
export const getProductByIdApi = async (productId) => {
  const response = await apiClient.get(`/api/products/${productId}`);
  return response.data;
};

/**
 * Create a new product.
 */
export const createProductApi = async (productData) => {
  const response = await apiClient.post('/api/products', productData);
  return response.data;
};

/**
 * Update an existing product by external product_id.
 */
export const updateProductApi = async (productId, productData) => {
  const response = await apiClient.put(`/api/products/${productId}`, productData);
  return response.data;
};

/**
 * Delete a product by external product_id.
 */
export const deleteProductApi = async (productId) => {
  const response = await apiClient.delete(`/api/products/${productId}`);
  return response.data;
};
