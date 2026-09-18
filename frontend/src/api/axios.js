import axios from 'axios';
import { AUTH_TOKEN_KEY, AUTH_USER_KEY } from '../utils/constants';

const baseURL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

const apiClient = axios.create({
  baseURL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

// Request interceptor to automatically inject Authorization token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem(AUTH_TOKEN_KEY);
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Helper to extract clean error message from FastAPI HTTP errors
export const extractErrorMessage = (error) => {
  if (!error) return 'An unexpected error occurred.';

  if (error.response) {
    const { status, data } = error.response;

    if (status === 422 && data?.detail) {
      if (Array.isArray(data.detail)) {
        const messages = data.detail.map(
          (err) => `${err.loc ? err.loc.join(' -> ') + ': ' : ''}${err.msg}`
        );
        return messages.join(' | ') || 'Invalid form input provided.';
      }
      return typeof data.detail === 'string' ? data.detail : 'Validation error occurred.';
    }

    if (data && data.detail) {
      return typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
    }

    switch (status) {
      case 400:
        return 'Bad request. Please check your request parameters.';
      case 401:
        return 'Session expired or invalid credentials. Please log in again.';
      case 403:
        return 'You do not have permission to perform this action.';
      case 404:
        return 'The requested resource was not found.';
      case 409:
        return 'A record with this identifier already exists.';
      case 500:
        return 'Internal server error. Please check backend service.';
      default:
        return `Request failed with status code ${status}.`;
    }
  }

  if (error.request) {
    return `Unable to connect to the backend server. Please verify the API service is reachable at ${baseURL}.`;
  }

  return error.message || 'An unknown error occurred.';
};

// Response interceptor to handle global auth failures
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      // Clear invalid token state
      localStorage.removeItem(AUTH_TOKEN_KEY);
      localStorage.removeItem(AUTH_USER_KEY);

      // Avoid redirect loops if already on login/register
      const currentPath = window.location.pathname;
      if (currentPath !== '/login' && currentPath !== '/register') {
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

export default apiClient;
