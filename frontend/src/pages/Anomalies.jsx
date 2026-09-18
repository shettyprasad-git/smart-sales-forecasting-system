import React, { useState, useEffect, useCallback } from 'react';
import { useLocation, useSearchParams } from 'react-router-dom';
import { getAnomaliesApi } from '../api/anomalies';
import { extractErrorMessage } from '../api/axios';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import EmptyState from '../components/EmptyState';
import InvestigationDrawer from '../components/InvestigationDrawer';
import {
  AlertTriangle,
  ShieldAlert,
  ArrowUpRight,
  ArrowDownRight,
  Filter,
  RefreshCw,
  Layers,
  Activity,
  ChevronLeft,
  ChevronRight,
  Search,
} from 'lucide-react';
import {
  formatCurrency,
  formatQuantity,
  formatDate,
} from '../utils/formatters';

const SEVERITY_CONFIG = {
  critical: {
    label: 'Critical',
    bg: 'bg-rose-500/10',
    border: 'border-rose-500/30',
    text: 'text-rose-400',
    dot: 'bg-rose-400',
  },
  high: {
    label: 'High',
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/30',
    text: 'text-amber-400',
    dot: 'bg-amber-400',
  },
  medium: {
    label: 'Medium',
    bg: 'bg-yellow-500/10',
    border: 'border-yellow-500/30',
    text: 'text-yellow-300',
    dot: 'bg-yellow-300',
  },
  low: {
    label: 'Low',
    bg: 'bg-sky-500/10',
    border: 'border-sky-500/30',
    text: 'text-sky-400',
    dot: 'bg-sky-400',
  },
};

const Anomalies = () => {
  const location = useLocation();
  const [searchParams] = useSearchParams();

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [anomaliesData, setAnomaliesData] = useState(null);
  const [selectedAnomalyId, setSelectedAnomalyId] = useState(null);

  useEffect(() => {
    const incomingId =
      location.state?.anomalyId ||
      searchParams.get('anomaly_id') ||
      searchParams.get('id');
    if (incomingId) {
      setSelectedAnomalyId(incomingId);
    }
  }, [location.state, searchParams]);

  // Filter states
  const [metricFilter, setMetricFilter] = useState('');
  const [severityFilter, setSeverityFilter] = useState('');
  const [directionFilter, setDirectionFilter] = useState('');
  const [entityType, setEntityType] = useState('aggregate');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [page, setPage] = useState(0);
  const pageSize = 20;

  const fetchAnomalies = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const params = {
        skip: page * pageSize,
        limit: pageSize,
        entity_type: entityType,
      };

      if (metricFilter) params.metric = metricFilter;
      if (severityFilter) params.severity = severityFilter;
      if (directionFilter) params.direction = directionFilter;
      if (startDate) params.start_date = startDate;
      if (endDate) params.end_date = endDate;

      const data = await getAnomaliesApi(params);
      setAnomaliesData(data);
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }, [metricFilter, severityFilter, directionFilter, entityType, startDate, endDate, page]);

  useEffect(() => {
    fetchAnomalies();
  }, [fetchAnomalies]);

  const handleResetFilters = () => {
    setMetricFilter('');
    setSeverityFilter('');
    setDirectionFilter('');
    setEntityType('aggregate');
    setStartDate('');
    setEndDate('');
    setPage(0);
  };

  const summary = anomaliesData?.summary;
  const items = anomaliesData?.items || [];
  const totalItems = anomaliesData?.total || 0;
  const totalPages = Math.ceil(totalItems / pageSize);

  return (
    <div className="space-y-8">
      {/* Top Banner & Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800/80 pb-6">
        <div>
          <div className="flex items-center space-x-2.5">
            <div className="p-2 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400">
              <ShieldAlert className="w-5 h-5" />
            </div>
            <h1 className="text-2xl font-extrabold text-slate-100 tracking-tight">
              Sales Anomaly Detection
            </h1>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Production statistical monitoring detecting unusual spikes and drops in sales volume and revenue.
          </p>
        </div>

        {/* Granularity Switcher */}
        <div className="flex items-center bg-slate-900 border border-slate-800 rounded-2xl p-1 shadow-inner self-start md:self-auto">
          <button
            onClick={() => {
              setEntityType('aggregate');
              setPage(0);
            }}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all cursor-pointer ${
              entityType === 'aggregate'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-950/50'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
          >
            System Aggregate
          </button>
          <button
            onClick={() => {
              setEntityType('category');
              setPage(0);
            }}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all cursor-pointer ${
              entityType === 'category'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-950/50'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
            }`}
          >
            By Category
          </button>
        </div>
      </div>

      {/* KPI Cards Grid */}
      {summary && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="rounded-2xl bg-slate-900 border border-slate-800 p-5 shadow-lg space-y-1">
            <div className="flex items-center justify-between text-slate-400 mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider">
                Total Flagged Anomalies
              </span>
              <AlertTriangle className="w-4 h-4 text-amber-400" />
            </div>
            <div className="text-2xl font-bold text-slate-100 font-mono">
              {summary.total_anomalies.toLocaleString()}
            </div>
            <p className="text-[11px] text-slate-400">
              {summary.anomaly_rate_percent}% of evaluated timeline
            </p>
          </div>

          <div className="rounded-2xl bg-slate-900 border border-slate-800 p-5 shadow-lg space-y-1">
            <div className="flex items-center justify-between text-slate-400 mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider">
                Critical Alerts
              </span>
              <ShieldAlert className="w-4 h-4 text-rose-400" />
            </div>
            <div className="text-2xl font-bold text-rose-400 font-mono">
              {summary.severity_breakdown.critical || 0}
            </div>
            <p className="text-[11px] text-slate-400">Extreme outliers (|z| ≥ 4.0)</p>
          </div>

          <div className="rounded-2xl bg-slate-900 border border-slate-800 p-5 shadow-lg space-y-1">
            <div className="flex items-center justify-between text-slate-400 mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider">
                High Severity Alerts
              </span>
              <Activity className="w-4 h-4 text-amber-400" />
            </div>
            <div className="text-2xl font-bold text-amber-400 font-mono">
              {summary.severity_breakdown.high || 0}
            </div>
            <p className="text-[11px] text-slate-400">Significant shifts (3.0 ≤ |z| &lt; 4.0)</p>
          </div>

          <div className="rounded-2xl bg-slate-900 border border-slate-800 p-5 shadow-lg space-y-1">
            <div className="flex items-center justify-between text-slate-400 mb-2">
              <span className="text-xs font-semibold uppercase tracking-wider">
                Spikes vs Drops
              </span>
              <Layers className="w-4 h-4 text-indigo-400" />
            </div>
            <div className="text-xl font-bold text-slate-100 font-mono flex items-center space-x-2">
              <span className="text-emerald-400">+{summary.direction_breakdown.spike || 0}</span>
              <span className="text-slate-600">/</span>
              <span className="text-rose-400">-{summary.direction_breakdown.drop || 0}</span>
            </div>
            <p className="text-[11px] text-slate-400">Positive vs negative deviations</p>
          </div>
        </div>
      )}

      {/* Filter & Search Bar */}
      <div className="rounded-2xl bg-slate-900/90 border border-slate-800 p-4 shadow-lg space-y-3">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800/60">
          <div className="flex items-center space-x-2 text-xs font-bold text-slate-300">
            <Filter className="w-3.5 h-3.5 text-indigo-400" />
            <span>Filter Anomaly Stream</span>
          </div>
          {(metricFilter || severityFilter || directionFilter || startDate || endDate) && (
            <button
              onClick={handleResetFilters}
              className="text-[11px] text-indigo-400 hover:text-indigo-300 transition-colors cursor-pointer"
            >
              Reset Filters
            </button>
          )}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {/* Metric Selector */}
          <div>
            <label className="block text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
              Metric
            </label>
            <select
              value={metricFilter}
              onChange={(e) => {
                setMetricFilter(e.target.value);
                setPage(0);
              }}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              <option value="">All Metrics</option>
              <option value="quantity">Daily Quantity (Units)</option>
              <option value="sales_amount">Sales Amount (Revenue)</option>
            </select>
          </div>

          {/* Severity Selector */}
          <div>
            <label className="block text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
              Severity
            </label>
            <select
              value={severityFilter}
              onChange={(e) => {
                setSeverityFilter(e.target.value);
                setPage(0);
              }}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              <option value="">All Severities</option>
              <option value="critical">Critical (|z| ≥ 4.0)</option>
              <option value="high">High (3.0 ≤ |z| &lt; 4.0)</option>
              <option value="medium">Medium (2.5 ≤ |z| &lt; 3.0)</option>
              <option value="low">Low (2.0 ≤ |z| &lt; 2.5)</option>
            </select>
          </div>

          {/* Direction Selector */}
          <div>
            <label className="block text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
              Direction
            </label>
            <select
              value={directionFilter}
              onChange={(e) => {
                setDirectionFilter(e.target.value);
                setPage(0);
              }}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              <option value="">All Directions</option>
              <option value="spike">Spikes Only (+)</option>
              <option value="drop">Drops Only (-)</option>
            </select>
          </div>

          {/* Start Date */}
          <div>
            <label className="block text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
              From Date
            </label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => {
                setStartDate(e.target.value);
                setPage(0);
              }}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
            />
          </div>

          {/* End Date */}
          <div>
            <label className="block text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1">
              To Date
            </label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => {
                setEndDate(e.target.value);
                setPage(0);
              }}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
            />
          </div>
        </div>
      </div>

      {/* Main Anomalies Table Section */}
      <div className="rounded-3xl bg-slate-900/90 border border-slate-800 p-6 shadow-xl backdrop-blur-sm space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
          <div>
            <h3 className="text-base font-bold text-slate-100 flex items-center space-x-2">
              <Activity className="w-4 h-4 text-indigo-400" />
              <span>Detected Anomaly Records</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Chronological register of statistical outliers with historical baselines and natural language explanations.
            </p>
          </div>

          <button
            onClick={fetchAnomalies}
            disabled={loading}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors cursor-pointer"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>

        {error ? (
          <ErrorMessage
            title="Failed to Load Anomalies"
            message={error}
            onRetry={fetchAnomalies}
          />
        ) : loading && items.length === 0 ? (
          <LoadingSpinner text="Computing robust anomaly baselines & querying outlier records..." size="lg" />
        ) : items.length === 0 ? (
          <EmptyState
            title="No Anomalies Found"
            description="No sales observations matched the selected filter criteria. Try expanding the date range or adjusting severity filters."
            icon={ShieldAlert}
          />
        ) : (
          <>
            <div className="rounded-2xl border border-slate-800 overflow-hidden">
              <div className="overflow-x-auto custom-scrollbar">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-950 sticky top-0 text-slate-400 font-semibold border-b border-slate-800 z-10">
                    <tr>
                      <th className="py-3 px-4">Date</th>
                      <th className="py-3 px-4">Entity</th>
                      <th className="py-3 px-4">Metric</th>
                      <th className="py-3 px-4">Severity</th>
                      <th className="py-3 px-4">Direction</th>
                      <th className="py-3 px-4 text-right">Actual Value</th>
                      <th className="py-3 px-4 text-right">Expected Baseline</th>
                      <th className="py-3 px-4 text-right">Deviation</th>
                      <th className="py-3 px-4 text-right">Anomaly Score</th>
                      <th className="py-3 px-4 max-w-xs">Explanation</th>
                      <th className="py-3 px-4 text-center">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-300">
                    {items.map((row) => {
                      const sevConfig = SEVERITY_CONFIG[row.severity] || SEVERITY_CONFIG.low;
                      const isSpike = row.direction === 'spike';
                      const isQuantity = row.metric === 'quantity';

                      return (
                        <tr
                          key={row.id}
                          className="hover:bg-slate-800/40 transition-colors"
                        >
                          <td className="py-3 px-4 font-mono font-medium text-slate-200 whitespace-nowrap">
                            {formatDate(row.date)}
                          </td>
                          <td className="py-3 px-4 whitespace-nowrap">
                            <span className="text-slate-300 font-semibold">
                              {row.entity_name || 'System Sales'}
                            </span>
                            {row.entity_type !== 'aggregate' && (
                              <span className="block text-[10px] text-slate-500 uppercase">
                                {row.entity_type} #{row.entity_id}
                              </span>
                            )}
                          </td>
                          <td className="py-3 px-4 whitespace-nowrap">
                            <span
                              className={`px-2 py-0.5 rounded-full text-[10px] font-semibold border ${
                                isQuantity
                                  ? 'bg-indigo-500/10 border-indigo-500/20 text-indigo-300'
                                  : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300'
                              }`}
                            >
                              {isQuantity ? 'Quantity' : 'Revenue'}
                            </span>
                          </td>
                          <td className="py-3 px-4 whitespace-nowrap">
                            <span
                              className={`inline-flex items-center space-x-1.5 px-2.5 py-0.5 rounded-full text-[10px] font-semibold border ${sevConfig.bg} ${sevConfig.border} ${sevConfig.text}`}
                            >
                              <span className={`w-1.5 h-1.5 rounded-full ${sevConfig.dot}`} />
                              <span>{sevConfig.label}</span>
                            </span>
                          </td>
                          <td className="py-3 px-4 whitespace-nowrap">
                            <div className="flex items-center space-x-1">
                              {isSpike ? (
                                <ArrowUpRight className="w-3.5 h-3.5 text-emerald-400" />
                              ) : (
                                <ArrowDownRight className="w-3.5 h-3.5 text-rose-400" />
                              )}
                              <span
                                className={`font-mono font-semibold ${
                                  isSpike ? 'text-emerald-400' : 'text-rose-400'
                                }`}
                              >
                                {isSpike ? `+${row.deviation_percent}%` : `${row.deviation_percent}%`}
                              </span>
                            </div>
                          </td>
                          <td className="py-3 px-4 text-right font-mono font-bold text-slate-100 whitespace-nowrap">
                            {isQuantity
                              ? formatQuantity(row.actual_value)
                              : formatCurrency(row.actual_value)}
                          </td>
                          <td className="py-3 px-4 text-right font-mono text-slate-400 whitespace-nowrap">
                            {isQuantity
                              ? formatQuantity(row.expected_value)
                              : formatCurrency(row.expected_value)}
                          </td>
                          <td
                            className={`py-3 px-4 text-right font-mono font-semibold whitespace-nowrap ${
                              isSpike ? 'text-emerald-400' : 'text-rose-400'
                            }`}
                          >
                            {isQuantity
                              ? `${isSpike ? '+' : ''}${row.deviation.toLocaleString()} units`
                              : formatCurrency(row.deviation)}
                          </td>
                          <td className="py-3 px-4 text-right font-mono text-indigo-300 font-bold whitespace-nowrap">
                            <span className="px-2 py-0.5 bg-indigo-950/60 border border-indigo-500/20 rounded-lg">
                              {row.anomaly_score.toFixed(2)}
                            </span>
                          </td>
                          <td className="py-3 px-4 text-slate-400 max-w-sm">
                            <div className="line-clamp-2 hover:line-clamp-none transition-all cursor-pointer text-[11px] leading-relaxed">
                              {row.explanation}
                            </div>
                          </td>
                          <td className="py-3 px-4 text-center whitespace-nowrap">
                            <button
                              onClick={() => setSelectedAnomalyId(row.id)}
                              className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-xl bg-indigo-600/10 hover:bg-indigo-600 text-indigo-400 hover:text-white border border-indigo-500/20 hover:border-indigo-600 text-[11px] font-semibold transition-all cursor-pointer shadow-xs"
                              title="Investigate root causes and driver attribution"
                            >
                              <Search className="w-3.5 h-3.5" />
                              <span>Investigate</span>
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Pagination Controls */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-2 text-xs text-slate-400">
              <div>
                Showing <strong className="text-slate-200">{items.length}</strong> of{' '}
                <strong className="text-slate-200">{totalItems.toLocaleString()}</strong> anomalies
                (Page {page + 1} of {Math.max(1, totalPages)})
              </div>

              <div className="flex items-center space-x-2">
                <button
                  onClick={() => setPage((prev) => Math.max(0, prev - 1))}
                  disabled={page === 0 || loading}
                  className="flex items-center space-x-1 px-3 py-1.5 rounded-xl border border-slate-800 bg-slate-950 text-slate-300 hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors cursor-pointer"
                >
                  <ChevronLeft className="w-4 h-4" />
                  <span>Previous</span>
                </button>

                <button
                  onClick={() => setPage((prev) => prev + 1)}
                  disabled={page + 1 >= totalPages || loading}
                  className="flex items-center space-x-1 px-3 py-1.5 rounded-xl border border-slate-800 bg-slate-950 text-slate-300 hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors cursor-pointer"
                >
                  <span>Next</span>
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </>
        )}
      </div>

      {/* Root-Cause Investigation Drawer / Modal */}
      <InvestigationDrawer
        anomalyId={selectedAnomalyId}
        isOpen={Boolean(selectedAnomalyId)}
        onClose={() => setSelectedAnomalyId(null)}
      />
    </div>
  );
};

export default Anomalies;
