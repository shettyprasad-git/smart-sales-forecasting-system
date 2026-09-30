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
      if (config.headers?.set) {
        config.headers.set('Authorization', `Bearer ${token}`);
      } else {
        config.headers = config.headers || {};
        config.headers.Authorization = `Bearer ${token}`;
        config.headers['Authorization'] = `Bearer ${token}`;
      }
    }

    // For multipart/form-data (FormData payload):
    // Do NOT manually force Content-Type: multipart/form-data without boundary.
    // Allow Axios and the browser to set Content-Type with the correct boundary parameter.
    if (typeof FormData !== 'undefined' && config.data instanceof FormData) {
      if (config.headers?.delete) {
        config.headers.delete('Content-Type');
      } else if (config.headers) {
        delete config.headers['Content-Type'];
        delete config.headers['content-type'];
      }
    }

    // Safe debugging telemetry (never log actual JWT, password, or refresh token)
    const hasAuth = !!(
      config.headers?.Authorization ||
      config.headers?.['Authorization'] ||
      (config.headers?.get && config.headers.get('Authorization'))
    );
    if (import.meta.env.DEV || import.meta.env.VITE_DEBUG_AUTH) {
      console.debug(
        `[apiClient] ${config.method?.toUpperCase()} ${config.url} auth_header_present=${hasAuth}`
      );
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

    // 401: Expired or invalid token
    if (status === 401) {
      return 'Your session has expired. Please sign in again.';
    }

    // 403: Forbidden
    if (status === 403) {
      if (data && data.detail && typeof data.detail === 'string') {
        return data.detail;
      }
      return 'You do not have permission.';
    }

    // 409: Conflict
    if (status === 409) {
      if (data && data.detail) {
        return typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
      }
      return 'A record with this identifier already exists.';
    }

    // 422: Validation response
    if (status === 422 && data?.detail) {
      if (Array.isArray(data.detail)) {
        const messages = data.detail.map(
          (err) => `${err.loc ? err.loc.join(' -> ') + ': ' : ''}${err.msg}`
        );
        return messages.join(' | ') || 'Validation error occurred.';
      }
      return typeof data.detail === 'string' ? data.detail : 'Validation error occurred.';
    }

    // 500: Internal server error
    if (status === 500) {
      return 'The backend encountered an internal error.';
    }

    // 502 / 503: Gateway / Service unavailable
    if (status === 502 || status === 503) {
      return 'Backend service temporarily unavailable.';
    }

    if (data && data.detail) {
      return typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
    }

    switch (status) {
      case 400:
        return 'Bad request. Please check your request parameters.';
      case 404:
        return 'The requested resource was not found.';
      default:
        return `Request failed with status code ${status}.`;
    }
  }

  // Network failure (no response received from backend)
  if (error.request) {
    return 'Unable to reach the backend server.';
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

      // Store notice for login screen
      sessionStorage.setItem('auth_notice', 'Your session has expired. Please sign in again.');

      // Avoid redirect loops if already on login/register
      const currentPath = window.location.pathname;
      if (currentPath !== '/login' && currentPath !== '/register') {
        window.location.href = '/login?expired=1';
      }
    }
    return Promise.reject(error);
  }
);

export default apiClient;
