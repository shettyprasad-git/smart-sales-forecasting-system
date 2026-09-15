import apiClient from './axios';

/**
 * Login user using OAuth2PasswordRequestForm (username & password).
 */
export const loginApi = async (email, password) => {
  const formData = new URLSearchParams();
  formData.append('username', email);
  formData.append('password', password);

  const response = await apiClient.post('/api/auth/login', formData, {
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
  });
  return response.data;
};

/**
 * Register a new user account.
 */
export const registerApi = async (email, password) => {
  const response = await apiClient.post('/api/auth/register', {
    email,
    password,
  });
  return response.data;
};

/**
 * Fetch current authenticated user profile.
 */
export const getMeApi = async () => {
  const response = await apiClient.get('/api/auth/me');
  return response.data;
};
