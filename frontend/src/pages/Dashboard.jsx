import React, { useState, useEffect, useCallback } from 'react';
import { getDashboardApi } from '../api/forecast';
import { extractErrorMessage } from '../api/axios';
import KpiCard from '../components/KpiCard';
import SalesChart from '../components/SalesChart';
import ForecastChart from '../components/ForecastChart';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import {
  Package,
  TrendingUp,
  DollarSign,
  Activity,
  Sparkles,
  Calendar,
  Cpu,
} from 'lucide-react';
import { formatCurrency, formatNumber } from '../utils/formatters';
import { FORECAST_HORIZONS, MODEL_DESCRIPTIONS } from '../utils/constants';

const Dashboard = () => {
  const [horizon, setHorizon] = useState(7);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dashboardData, setDashboardData] = useState(null);

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

  useEffect(() => {
    fetchDashboard(horizon);
  }, [horizon, fetchDashboard]);

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

      {error ? (
        <ErrorMessage
          title="Failed to Load Executive Dashboard"
          message={error}
          onRetry={() => fetchDashboard(horizon)}
        />
      ) : loading && !dashboardData ? (
        <LoadingSpinner text="Connecting to ML backend & computing dashboard telemetry..." size="lg" />
      ) : (
        <>
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
              icon={DollarSign}
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
                  <DollarSign className="w-4 h-4 text-emerald-400" />
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
