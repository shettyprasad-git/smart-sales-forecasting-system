/**
 * Utility functions for formatting values cleanly in the dashboard UI.
 */

/**
 * Format currency in Indian Rupees (₹) with smart compact notation for large numbers.
 * Examples: ₹7.12B, ₹245.6M, ₹15,000, ₹949.50
 */
export const formatCurrency = (value) => {
  if (value === null || value === undefined || isNaN(value)) {
    return '₹0';
  }

  const num = Number(value);
  const absNum = Math.abs(num);

  if (absNum >= 1_000_000_000) {
    return `₹${(num / 1_000_000_000).toFixed(2)}B`;
  }
  if (absNum >= 1_000_000) {
    return `₹${(num / 1_000_000).toFixed(2)}M`;
  }
  if (absNum >= 100_000) {
    return `₹${(num / 100_000).toFixed(2)}L`;
  }

  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 2,
  }).format(num);
};

/**
 * Format quantity values with commas and clean decimal handling.
 * Examples: 4,938 units or 5,000
 */
export const formatQuantity = (value, unit = 'units') => {
  if (value === null || value === undefined || isNaN(value)) {
    return `0 ${unit}`.trim();
  }

  const num = Number(value);
  const formatted = new Intl.NumberFormat('en-IN', {
    maximumFractionDigits: 1,
  }).format(num);

  return unit ? `${formatted} ${unit}` : formatted;
};

/**
 * Format raw numbers with commas.
 */
export const formatNumber = (value) => {
  if (value === null || value === undefined || isNaN(value)) {
    return '0';
  }
  return new Intl.NumberFormat('en-IN', {
    maximumFractionDigits: 2,
  }).format(Number(value));
};

/**
 * Format ISO or YYYY-MM-DD date string into readable date.
 * Example: '2026-09-15' -> '15 Sep 2026'
 */
export const formatDate = (dateStr) => {
  if (!dateStr) return '';
  try {
    const date = new Date(dateStr);
    if (isNaN(date.getTime())) return dateStr;
    return new Intl.DateTimeFormat('en-GB', {
      day: 'numeric',
      month: 'short',
      year: 'numeric',
    }).format(date);
  } catch {
    return dateStr;
  }
};

/**
 * Format percentages.
 * Example: 5 -> '5.0%'
 */
export const formatPercent = (value) => {
  if (value === null || value === undefined || isNaN(value)) {
    return '0.0%';
  }
  return `${Number(value).toFixed(1)}%`;
};
