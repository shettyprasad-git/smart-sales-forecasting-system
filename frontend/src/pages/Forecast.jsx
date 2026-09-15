import React, { useState, useEffect, useCallback } from 'react';
import { getForecastApi, getForecastHistoryApi } from '../api/forecast';
import { extractErrorMessage } from '../api/axios';
import ForecastChart from '../components/ForecastChart';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import {
  TrendingUp,
  Cpu,
  Calendar,
  Sparkles,
  ArrowUpRight,
  ArrowDownRight,
  Layers,
  Activity,
  Award,
} from 'lucide-react';
import { formatNumber, formatDate } from '../utils/formatters';
import { FORECAST_HORIZONS, MODEL_DESCRIPTIONS } from '../utils/constants';

const Forecast = () => {
  const [horizon, setHorizon] = useState(7);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [forecastResponse, setForecastResponse] = useState(null);
  const [historicalData, setHistoricalData] = useState([]);

  const fetchForecast = useCallback(async (selectedHorizon) => {
    try {
      setLoading(true);
      setError(null);

      // Fetch ML forecast and recent historical data concurrently
      const [forecastRes, historyRes] = await Promise.all([
        getForecastApi(selectedHorizon),
        getForecastHistoryApi(90),
      ]);

      setForecastResponse(forecastRes);
      setHistoricalData(historyRes.records || []);
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchForecast(horizon);
  }, [horizon, fetchForecast]);

  const handleHorizonChange = (hVal) => {
    if (hVal !== horizon) {
      setHorizon(hVal);
    }
  };

  const forecastPoints = forecastResponse?.forecast || [];
  const modelName = forecastResponse?.model || 'ML Model';

  // Derived Insights Calculations
  const totalPredictedDemand = forecastPoints.reduce((acc, curr) => acc + curr.predicted_quantity, 0);
  const avgPredictedDemand = forecastPoints.length > 0 ? totalPredictedDemand / forecastPoints.length : 0;

  // Find Peak and Lowest Demand Days
  let peakDay = null;
  let lowestDay = null;
  if (forecastPoints.length > 0) {
    peakDay = forecastPoints.reduce((prev, curr) =>
      curr.predicted_quantity > prev.predicted_quantity ? curr : prev
    );
    lowestDay = forecastPoints.reduce((prev, curr) =>
      curr.predicted_quantity < prev.predicted_quantity ? curr : prev
    );
  }

  // Calculate percentage shift vs historical average demand
  let demandTrendPercent = 0;
  if (historicalData.length > 0 && avgPredictedDemand > 0) {
    const totalHistQty = historicalData.reduce((acc, curr) => acc + curr.quantity, 0);
    const avgHistQty = totalHistQty / historicalData.length;
    if (avgHistQty > 0) {
      demandTrendPercent = ((avgPredictedDemand - avgHistQty) / avgHistQty) * 100;
    }
  }

  return (
    <div className="space-y-8">
      {/* Top Banner Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800/80 pb-6">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-100 tracking-tight">
            Demand Forecast Engine
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Machine learning forecast models analyzing seasonality, trends, and future demand horizons.
          </p>
        </div>

        {/* Horizon Selector Buttons */}
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
          title="Forecast Generation Failed"
          message={error}
          onRetry={() => fetchForecast(horizon)}
        />
      ) : loading && !forecastResponse ? (
        <LoadingSpinner text={`Executing ML model for ${horizon}-day horizon...`} size="lg" />
      ) : (
        <>
          {/* Active Model Summary Banner */}
          <div className="rounded-3xl bg-gradient-to-r from-slate-900 via-indigo-950/30 to-slate-900 border border-slate-800 p-6 shadow-xl flex flex-col md:flex-row md:items-center justify-between gap-6 backdrop-blur-sm">
            <div className="flex items-start space-x-4">
              <div className="p-3 bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 rounded-2xl">
                <Cpu className="w-6 h-6" />
              </div>
              <div className="space-y-1">
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                    Assigned Model:
                  </span>
                  <span className="px-2.5 py-0.5 rounded-lg bg-indigo-500/20 border border-indigo-500/30 text-indigo-300 text-xs font-bold font-mono">
                    {modelName}
                  </span>
                </div>
                <h3 className="text-base font-bold text-slate-100">
                  {horizon}-Day Demand Prediction
                </h3>
                <p className="text-xs text-slate-400 max-w-xl">
                  {MODEL_DESCRIPTIONS[modelName] ||
                    'Production Machine Learning forecasting model trained on sales historical telemetry.'}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-4 bg-slate-950/60 p-4 rounded-2xl border border-slate-800 self-start md:self-auto">
              <div className="text-center">
                <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
                  Forecast Period
                </div>
                <div className="text-sm font-bold text-slate-200 font-mono mt-0.5">
                  {horizon} Days
                </div>
              </div>
              <div className="h-8 w-px bg-slate-800" />
              <div className="text-center">
                <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">
                  Data Points
                </div>
                <div className="text-sm font-bold text-slate-200 font-mono mt-0.5">
                  {forecastPoints.length}
                </div>
              </div>
            </div>
          </div>

          {/* Derived Analytics KPI Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="rounded-2xl bg-slate-900 border border-slate-800 p-5 shadow-lg space-y-1">
              <div className="flex items-center justify-between text-slate-400 mb-2">
                <span className="text-xs font-semibold uppercase tracking-wider">
                  Total Predicted Demand
                </span>
                <Layers className="w-4 h-4 text-indigo-400" />
              </div>
              <div className="text-2xl font-bold text-slate-100 font-mono">
                {formatNumber(totalPredictedDemand)} units
              </div>
              <p className="text-[11px] text-slate-400">Sum for upcoming {horizon} days</p>
            </div>

            <div className="rounded-2xl bg-slate-900 border border-slate-800 p-5 shadow-lg space-y-1">
              <div className="flex items-center justify-between text-slate-400 mb-2">
                <span className="text-xs font-semibold uppercase tracking-wider">
                  Avg Daily Prediction
                </span>
                <Activity className="w-4 h-4 text-sky-400" />
              </div>
              <div className="text-2xl font-bold text-slate-100 font-mono">
                {formatNumber(avgPredictedDemand)} units/day
              </div>
              <p className="text-[11px] text-slate-400">Mean forecasted daily volume</p>
            </div>

            <div className="rounded-2xl bg-slate-900 border border-slate-800 p-5 shadow-lg space-y-1">
              <div className="flex items-center justify-between text-slate-400 mb-2">
                <span className="text-xs font-semibold uppercase tracking-wider">
                  Peak Demand Day
                </span>
                <ArrowUpRight className="w-4 h-4 text-emerald-400" />
              </div>
              <div className="text-lg font-bold text-emerald-400 font-mono truncate">
                {peakDay ? `${formatNumber(peakDay.predicted_quantity)} units` : 'N/A'}
              </div>
              <p className="text-[11px] text-slate-400">
                {peakDay ? formatDate(peakDay.date) : 'No peak date'}
              </p>
            </div>

            <div className="rounded-2xl bg-slate-900 border border-slate-800 p-5 shadow-lg space-y-1">
              <div className="flex items-center justify-between text-slate-400 mb-2">
                <span className="text-xs font-semibold uppercase tracking-wider">
                  Demand Shift vs Hist.
                </span>
                <TrendingUp className="w-4 h-4 text-amber-400" />
              </div>
              <div className={`text-2xl font-bold font-mono ${demandTrendPercent >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {demandTrendPercent >= 0 ? `+${demandTrendPercent.toFixed(1)}%` : `${demandTrendPercent.toFixed(1)}%`}
              </div>
              <p className="text-[11px] text-slate-400">Compared to recent daily mean</p>
            </div>
          </div>

          {/* Interactive Chart Section */}
          <div className="rounded-3xl bg-slate-900/90 border border-slate-800 p-6 shadow-xl backdrop-blur-sm space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
              <div>
                <h3 className="text-base font-bold text-slate-100 flex items-center space-x-2">
                  <TrendingUp className="w-4 h-4 text-indigo-400" />
                  <span>Visual Demand Trajectory</span>
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Historical telemetry integrated with {horizon}-day {modelName} ML forecast curve.
                </p>
              </div>
            </div>

            {loading ? (
              <LoadingSpinner text="Re-calculating visualization curve..." />
            ) : (
              <ForecastChart historical={historicalData} forecast={forecastPoints} modelName={modelName} />
            )}
          </div>

          {/* Forecast Data Table */}
          <div className="rounded-3xl bg-slate-900/90 border border-slate-800 p-6 shadow-xl backdrop-blur-sm space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
              <div>
                <h3 className="text-base font-bold text-slate-100 flex items-center space-x-2">
                  <Sparkles className="w-4 h-4 text-amber-400" />
                  <span>Predicted Daily Demand Breakdown</span>
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Tabular list of exact date-by-date quantity predictions generated by {modelName}.
                </p>
              </div>
            </div>

            <div className="rounded-2xl border border-slate-800 overflow-hidden">
              <div className="overflow-x-auto custom-scrollbar max-h-96">
                <table className="w-full text-left text-xs">
                  <thead className="bg-slate-950 sticky top-0 text-slate-400 font-semibold border-b border-slate-800 z-10">
                    <tr>
                      <th className="py-3 px-4">#</th>
                      <th className="py-3 px-4">Forecast Date</th>
                      <th className="py-3 px-4 text-right">Predicted Quantity (Units)</th>
                      <th className="py-3 px-4 text-right">Status / Variance</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-slate-300">
                    {forecastPoints.map((row, idx) => {
                      const isPeak = peakDay && row.date === peakDay.date;
                      const isLowest = lowestDay && row.date === lowestDay.date;

                      return (
                        <tr key={row.date} className="hover:bg-slate-800/40 transition-colors">
                          <td className="py-3 px-4 font-mono text-slate-500">{idx + 1}</td>
                          <td className="py-3 px-4 font-mono font-medium text-slate-200">
                            {formatDate(row.date)}
                          </td>
                          <td className="py-3 px-4 text-right font-mono font-bold text-indigo-300">
                            {formatNumber(row.predicted_quantity)}
                          </td>
                          <td className="py-3 px-4 text-right">
                            {isPeak ? (
                              <span className="px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[10px] font-semibold">
                                Peak Day
                              </span>
                            ) : isLowest ? (
                              <span className="px-2 py-0.5 rounded-full bg-rose-500/10 border border-rose-500/20 text-rose-400 text-[10px] font-semibold">
                                Lowest Day
                              </span>
                            ) : (
                              <span className="text-[10px] text-slate-500">Normal Variance</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default Forecast;
