import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { getDashboardApi } from '../api/forecast';
import { getMonitoringSummary } from '../api/monitoring';
import { listDecisionsApi } from '../api/decisions';
import { getCurrentDatasetApi } from '../api/datasets';
import { extractErrorMessage } from '../api/axios';
import KpiCard from '../components/KpiCard';
import SalesChart from '../components/SalesChart';
import ForecastChart from '../components/ForecastChart';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import {
  Package,
  TrendingUp,
  IndianRupee,
  Activity,
  Sparkles,
  Calendar,
  Cpu,
  ShieldAlert,
  ShieldCheck,
  Sliders,
  Bell,
  ArrowRight,
  Database,
} from 'lucide-react';
import { formatCurrency, formatNumber } from '../utils/formatters';
import { FORECAST_HORIZONS, MODEL_DESCRIPTIONS } from '../utils/constants';

const Dashboard = () => {
  const navigate = useNavigate();
  const [horizon, setHorizon] = useState(7);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dashboardData, setDashboardData] = useState(null);

  // Intelligence Platform Telemetry
  const [intelSummary, setIntelSummary] = useState(null);
  const [pendingDecisionsCount, setPendingDecisionsCount] = useState(0);
  const [intelLoading, setIntelLoading] = useState(true);
  const [intelError, setIntelError] = useState(false);
  const [decisionsError, setDecisionsError] = useState(false);
  const [currentDataset, setCurrentDataset] = useState(null);

  const fetchCurrentDataset = useCallback(async () => {
    try {
      const data = await getCurrentDatasetApi();
      setCurrentDataset(data);
    } catch {
      setCurrentDataset(null);
    }
  }, []);

  const fetchDashboard = useCallback(async (selectedHorizon) => {
    try {
      setLoading(true);
      setError(null);
      const data = await getDashboardApi(selectedHorizon, 365);
      setDashboardData(data);
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchIntelligenceData = useCallback(async () => {
    setIntelLoading(true);
    setIntelError(false);
    setDecisionsError(false);
    try {
      const [summaryRes, decisionsRes] = await Promise.allSettled([
        getMonitoringSummary(),
        listDecisionsApi({ status: 'pending_review' }),
      ]);
      if (summaryRes.status === 'fulfilled') {
        setIntelSummary(summaryRes.value);
      } else {
        setIntelError(true);
      }
      if (decisionsRes.status === 'fulfilled') {
        setPendingDecisionsCount(
          Array.isArray(decisionsRes.value) ? decisionsRes.value.length : 0
        );
      } else {
        setDecisionsError(true);
      }
    } catch {
      setIntelError(true);
      setDecisionsError(true);
    } finally {
      setIntelLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDashboard(horizon);
    fetchIntelligenceData();
    fetchCurrentDataset();
  }, [horizon, fetchDashboard, fetchIntelligenceData, fetchCurrentDataset]);

  const handleHorizonChange = (newHorizon) => {
    if (newHorizon !== horizon) {
      setHorizon(newHorizon);
    }
  };

  const kpis = dashboardData?.kpis || {};
  const historical = dashboardData?.historical || [];
  const forecast = dashboardData?.forecast || [];
  const modelName = dashboardData?.forecast_model || 'ML Model';

  return (
    <div className="space-y-8">
      {/* Top Banner & Page Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800/80 pb-6">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-100 tracking-tight">
            Sales Forecasting Dashboard
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Monitor historical performance metrics and analyze ML-driven future demand predictions.
          </p>
        </div>

        {/* Forecast Horizon Tabs */}
        <div className="flex items-center bg-slate-900 border border-slate-800 rounded-2xl p-1 shadow-inner self-start md:self-auto">
          <span className="text-[11px] font-semibold text-slate-400 px-3 flex items-center space-x-1.5 hidden sm:flex">
            <Calendar className="w-3.5 h-3.5 text-indigo-400" />
            <span>Horizon:</span>
          </span>
          {FORECAST_HORIZONS.map((h) => {
            const isActive = horizon === h.value;
            return (
              <button
                key={h.value}
                onClick={() => handleHorizonChange(h.value)}
                className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all duration-150 cursor-pointer ${
                  isActive
                    ? 'bg-indigo-600 text-white shadow-md shadow-indigo-950/50'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                {h.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Active Dataset Status Strip */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 px-4 py-3 rounded-2xl bg-slate-900/80 border border-slate-800 text-xs">
        <div className="flex items-center space-x-2.5">
          <Database className={`w-4 h-4 ${currentDataset ? 'text-emerald-400' : 'text-amber-400'}`} />
          <div>
            <span className="font-semibold text-slate-200">
              {currentDataset ? `Active Dataset: ${currentDataset.filename || currentDataset.original_filename}` : 'Using Baseline Repository Dataset'}
            </span>
            <span className="text-slate-400 ml-2 hidden sm:inline">
              {currentDataset
                ? `(${formatNumber(currentDataset.rows_imported ?? currentDataset.row_count ?? 0)} records, ${currentDataset.start_date || currentDataset.min_date} → ${currentDataset.end_date || currentDataset.max_date})`
                : '• Upload your own sales CSV to switch predictions and intelligence to your tenant data.'}
            </span>
          </div>
        </div>
        <button
          onClick={() => navigate('/datasets')}
          className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 flex items-center space-x-1 cursor-pointer self-start sm:self-auto"
        >
          <span>{currentDataset ? 'Switch / Manage' : 'Upload Dataset'}</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
      </div>

      {error ? (
        <ErrorMessage
          title="Failed to Load Executive Dashboard"
          message={error}
          onRetry={() => {
            fetchDashboard(horizon);
            fetchIntelligenceData();
          }}
        />
      ) : loading && !dashboardData ? (
        <LoadingSpinner text="Connecting to ML backend & computing dashboard telemetry..." size="lg" />
      ) : (!historical.length && !forecast.length) ? (
        <div className="rounded-3xl bg-slate-900/90 border border-slate-800 p-12 text-center shadow-xl space-y-4">
          <div className="inline-flex p-4 rounded-full bg-slate-800/80 text-slate-400 mb-2">
            <Package className="w-8 h-8" />
          </div>
          <h3 className="text-lg font-bold text-slate-200">No Sales Data Available</h3>
          <p className="text-sm text-slate-400 max-w-md mx-auto">
            No sales records or predictions were found for this horizon. Please upload or seed a dataset to view analytics.
          </p>
        </div>
      ) : (
        <>
          {/* Executive Financial Intelligence & Proactive Governance Section */}
          <div className="rounded-3xl bg-gradient-to-b from-slate-900 via-slate-900/90 to-slate-950 border border-slate-800/90 p-6 shadow-2xl space-y-5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800/80 pb-4">
              <div>
                <div className="flex items-center space-x-2 text-xs font-bold text-indigo-400 uppercase tracking-wider">
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Executive Financial Intelligence & Proactive Monitoring</span>
                </div>
                <h2 className="text-lg font-extrabold text-slate-100 tracking-tight mt-0.5">
                  Real-Time Intelligence & Human Decision Governance
                </h2>
                <p className="text-xs text-slate-400">
                  Continuous statistical scanning across demand anomalies, prescriptive actions, and human-in-the-loop approvals.
                </p>
              </div>

              {/* Quick Jump Links */}
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => navigate('/intelligence')}
                  className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-indigo-600/10 hover:bg-indigo-600/20 text-indigo-300 border border-indigo-500/30 transition-all cursor-pointer"
                >
                  <Bell className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Alert Feed</span>
                  <ArrowRight className="w-3 h-3" />
                </button>
                <button
                  type="button"
                  onClick={() => navigate('/decisions')}
                  className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-amber-600/10 hover:bg-amber-600/20 text-amber-300 border border-amber-500/30 transition-all cursor-pointer"
                >
                  <ShieldCheck className="w-3.5 h-3.5 text-amber-400" />
                  <span>Decisions</span>
                  {pendingDecisionsCount > 0 && (
                    <span className="px-1.5 py-0.2 rounded-full bg-amber-500 text-slate-950 font-mono text-[10px] font-bold">
                      {pendingDecisionsCount}
                    </span>
                  )}
                </button>
                <button
                  type="button"
                  onClick={() => navigate('/simulation')}
                  className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-emerald-600/10 hover:bg-emerald-600/20 text-emerald-300 border border-emerald-500/30 transition-all cursor-pointer"
                >
                  <Sliders className="w-3.5 h-3.5 text-emerald-400" />
                  <span>What-If</span>
                </button>
              </div>
            </div>

            {/* Intelligence KPI Cards Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {/* Active Alerts */}
              <div
                onClick={() => navigate('/intelligence')}
                className="rounded-2xl bg-slate-950/70 border border-slate-800 hover:border-indigo-500/40 p-4 transition-all duration-200 cursor-pointer group shadow-inner"
              >
                <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
                  <span className="font-semibold uppercase tracking-wider text-[11px] flex items-center space-x-1.5">
                    <Bell className="w-3.5 h-3.5 text-rose-400" />
                    <span>Unresolved Alerts</span>
                  </span>
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-rose-500/10 text-rose-400 border border-rose-500/20">
                    {intelSummary?.critical_alerts ?? 0} Critical
                  </span>
                </div>
                <div className="text-2xl font-black text-slate-100 font-mono tracking-tight group-hover:text-indigo-300 transition-colors">
                  {intelLoading ? (
                    <span className="text-base text-slate-500 font-sans animate-pulse">Loading...</span>
                  ) : intelError ? (
                    <span className="text-xs text-rose-400 font-sans font-medium">Unavailable</span>
                  ) : (
                    intelSummary?.unresolved_alerts ?? '0'
                  )}
                </div>
                <div className="flex items-center justify-between text-[11px] text-slate-400 mt-2 pt-2 border-t border-slate-800/80">
                  <span>
                    {intelError ? 'Telemetry offline' : `${intelSummary?.new_alerts ?? 0} newly surfaced`}
                  </span>
                  <span className="text-indigo-400 group-hover:translate-x-0.5 transition-transform flex items-center">
                    Review <ArrowRight className="w-3 h-3 ml-0.5" />
                  </span>
                </div>
              </div>

              {/* Pending Decisions */}
              <div
                onClick={() => navigate('/decisions')}
                className="rounded-2xl bg-slate-950/70 border border-slate-800 hover:border-amber-500/40 p-4 transition-all duration-200 cursor-pointer group shadow-inner"
              >
                <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
                  <span className="font-semibold uppercase tracking-wider text-[11px] flex items-center space-x-1.5">
                    <ShieldCheck className="w-3.5 h-3.5 text-amber-400" />
                    <span>Governance Reviews</span>
                  </span>
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-500/10 text-amber-300 border border-amber-500/20">
                    Human Gate
                  </span>
                </div>
                <div className="text-2xl font-black text-slate-100 font-mono tracking-tight group-hover:text-amber-300 transition-colors">
                  {intelLoading ? (
                    <span className="text-base text-slate-500 font-sans animate-pulse">Loading...</span>
                  ) : decisionsError ? (
                    <span className="text-xs text-amber-400 font-sans font-medium">Unavailable</span>
                  ) : (
                    pendingDecisionsCount
                  )}
                </div>
                <div className="flex items-center justify-between text-[11px] text-slate-400 mt-2 pt-2 border-t border-slate-800/80">
                  <span>{decisionsError ? 'Telemetry offline' : 'Awaiting human approval'}</span>
                  <span className="text-amber-400 group-hover:translate-x-0.5 transition-transform flex items-center">
                    Decide <ArrowRight className="w-3 h-3 ml-0.5" />
                  </span>
                </div>
              </div>

              {/* Detected Anomalies */}
              <div
                onClick={() => navigate('/anomalies')}
                className="rounded-2xl bg-slate-950/70 border border-slate-800 hover:border-indigo-500/40 p-4 transition-all duration-200 cursor-pointer group shadow-inner"
              >
                <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
                  <span className="font-semibold uppercase tracking-wider text-[11px] flex items-center space-x-1.5">
                    <ShieldAlert className="w-3.5 h-3.5 text-indigo-400" />
                    <span>Sales Anomalies</span>
                  </span>
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-indigo-500/10 text-indigo-300 border border-indigo-500/20">
                    Statistical
                  </span>
                </div>
                <div className="text-2xl font-black text-slate-100 font-mono tracking-tight group-hover:text-indigo-300 transition-colors">
                  {intelSummary?.total_anomalies ?? 'Active'}
                </div>
                <div className="flex items-center justify-between text-[11px] text-slate-400 mt-2 pt-2 border-t border-slate-800/80">
                  <span>Root-cause telemetry</span>
                  <span className="text-indigo-400 group-hover:translate-x-0.5 transition-transform flex items-center">
                    Investigate <ArrowRight className="w-3 h-3 ml-0.5" />
                  </span>
                </div>
              </div>

              {/* What-If Simulation Sandbox */}
              <div
                onClick={() => navigate('/simulation')}
                className="rounded-2xl bg-slate-950/70 border border-slate-800 hover:border-emerald-500/40 p-4 transition-all duration-200 cursor-pointer group shadow-inner"
              >
                <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
                  <span className="font-semibold uppercase tracking-wider text-[11px] flex items-center space-x-1.5">
                    <Sliders className="w-3.5 h-3.5 text-emerald-400" />
                    <span>What-If Sandbox</span>
                  </span>
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
                    Hypothetical
                  </span>
                </div>
                <div className="text-2xl font-black text-emerald-400 font-mono tracking-tight group-hover:text-emerald-300 transition-colors">
                  Simulator
                </div>
                <div className="flex items-center justify-between text-[11px] text-slate-400 mt-2 pt-2 border-t border-slate-800/80">
                  <span>Counterfactual scenarios</span>
                  <span className="text-emerald-400 group-hover:translate-x-0.5 transition-transform flex items-center">
                    Simulate <ArrowRight className="w-3 h-3 ml-0.5" />
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* KPI Cards Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <KpiCard
              title="Total Historical Volume"
              value={`${formatNumber(kpis.total_historical_quantity || 0)} units`}
              icon={Package}
              description="Aggregated lifetime demand units"
              color="sky"
            />
            <KpiCard
              title="Total Sales Revenue"
              value={formatCurrency(kpis.total_historical_sales || 0)}
              icon={IndianRupee}
              description="Gross historical sales amount"
              color="indigo"
            />
            <KpiCard
              title="Total Net Profit"
              value={formatCurrency(kpis.total_historical_profit || 0)}
              icon={TrendingUp}
              description="Historical cumulative net profit"
              color="emerald"
            />
            <KpiCard
              title="Avg Daily Demand"
              value={`${formatNumber(kpis.average_daily_quantity || 0)} units`}
              icon={Activity}
              description="Mean quantity sold per day"
              color="amber"
            />
          </div>

          {/* Model Status Bar */}
          <div className="rounded-2xl bg-gradient-to-r from-indigo-950/40 via-slate-900 to-slate-900 border border-indigo-500/20 p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 shadow-lg">
            <div className="flex items-center space-x-3">
              <div className="p-2.5 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
                <Cpu className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-bold text-slate-200">Active Production Model:</span>
                  <span className="px-2 py-0.5 rounded-md bg-indigo-500/20 border border-indigo-500/30 text-indigo-300 text-xs font-semibold font-mono">
                    {modelName}
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  {MODEL_DESCRIPTIONS[modelName] || 'Machine learning model optimized for target horizon.'}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-2 text-xs text-slate-400 bg-slate-950/60 px-3 py-1.5 rounded-xl border border-slate-800 self-start sm:self-auto">
              <Sparkles className="w-3.5 h-3.5 text-amber-400 animate-pulse" />
              <span>Horizon: <strong className="text-slate-200">{horizon} Days</strong></span>
            </div>
          </div>

          {/* Forecast Demand Chart Card */}
          <div className="rounded-3xl bg-slate-900/90 border border-slate-800 p-6 shadow-xl backdrop-blur-sm space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
              <div>
                <h3 className="text-base font-bold text-slate-100 flex items-center space-x-2">
                  <TrendingUp className="w-4 h-4 text-indigo-400" />
                  <span>Demand Forecast & Historical Comparison</span>
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Seamless visual continuity between past demand volume and {horizon}-day predictions.
                </p>
              </div>
            </div>

            {loading ? (
              <LoadingSpinner text="Updating forecast telemetry..." />
            ) : (
              <ForecastChart historical={historical} forecast={forecast} modelName={modelName} />
            )}
          </div>

          {/* Financial Revenue & Profit Chart Card */}
          <div className="rounded-3xl bg-slate-900/90 border border-slate-800 p-6 shadow-xl backdrop-blur-sm space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
              <div>
                <h3 className="text-base font-bold text-slate-100 flex items-center space-x-2">
                  <IndianRupee className="w-4 h-4 text-emerald-400" />
                  <span>Historical Revenue & Profit Trend</span>
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Financial performance analysis detailing historical gross revenue vs net profit.
                </p>
              </div>
            </div>

            <SalesChart data={historical} />
          </div>
        </>
      )}
    </div>
  );
};

export default Dashboard;
