/**
 * Application constants and ML model horizon mappings.
 */

export const AUTH_TOKEN_KEY = 'smart_sales_access_token';
export const AUTH_USER_KEY = 'smart_sales_user';

export const FORECAST_HORIZONS = [
  { value: 7, label: '7 Days', model: 'Random Forest' },
  { value: 30, label: '30 Days', model: 'Gradient Boosting' },
  { value: 90, label: '90 Days', model: 'Linear Regression' },
];

export const MODEL_DESCRIPTIONS = {
  'Random Forest': 'Short-term non-linear pattern & holiday spike detection model.',
  'Gradient Boosting': 'Medium-term boosted ensemble trend forecasting model.',
  'Linear Regression': 'Long-term baseline macro trend forecasting model.',
};
