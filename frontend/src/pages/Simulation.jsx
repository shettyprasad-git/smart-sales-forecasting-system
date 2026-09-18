import React, { useState, useEffect } from 'react';
import { useSearchParams, useLocation, useNavigate } from 'react-router-dom';
import {
  Sliders,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle2,
  BrainCircuit,
  Info,
  ShieldAlert,
  ShieldCheck,
  ArrowUpRight,
  ArrowDownRight,
  Layers,
  Sparkles,
  Calendar,
  IndianRupee,
  Activity,
  FileText,
  ArrowRight,
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';

import SimulationPanel from '../components/SimulationPanel';
import { runSimulationApi } from '../api/simulations';
import { createDecisionApi } from '../api/decisions';
import { formatCurrency, formatQuantity, formatDate, formatPercent, formatNumber } from '../utils/formatters';

const CustomChartTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const isPositive = data.delta_quantity >= 0;

    return (
      <div className="bg-slate-900 border border-slate-700/80 p-3.5 rounded-xl shadow-2xl text-xs space-y-2 min-w-[220px]">
        <div className="flex items-center justify-between border-b border-slate-800 pb-1.5">
          <span className="font-semibold text-slate-200">{formatDate(label)}</span>
          <span
            className={`font-mono text-[11px] font-bold ${
              isPositive ? 'text-emerald-400' : 'text-rose-400'
            }`}
          >
            {isPositive ? `+${data.delta_percent}%` : `${data.delta_percent}%`}
          </span>
        </div>

        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <span className="flex items-center space-x-1.5 text-indigo-400">
              <span className="w-2 h-2 rounded-full bg-indigo-500" />
              <span>Baseline:</span>
            </span>
            <span className="font-mono font-bold text-slate-100">
              {formatNumber(data.baseline_quantity)} units
            </span>
          </div>

          <div className="flex items-center justify-between">
            <span className="flex items-center space-x-1.5 text-emerald-400">
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
              <span>Scenario:</span>
            </span>
            <span className="font-mono font-bold text-slate-100">
              {formatNumber(data.scenario_quantity)} units
            </span>
          </div>

          <div className="flex items-center justify-between pt-1 border-t border-slate-800/60">
            <span className="text-slate-400">Volume Variance:</span>
            <span
              className={`font-mono font-bold ${
                isPositive ? 'text-emerald-400' : 'text-rose-400'
              }`}
            >
              {isPositive ? `+${formatNumber(data.delta_quantity)}` : formatNumber(data.delta_quantity)} units
            </span>
          </div>

          {data.scenario_revenue !== null && data.scenario_revenue !== undefined && (
            <div className="flex items-center justify-between text-[11px]">
              <span className="text-slate-400">Scenario Revenue:</span>
              <span className="font-mono font-semibold text-slate-200">
                {formatCurrency(data.scenario_revenue)}
              </span>
            </div>
          )}
        </div>
      </div>
    );
  }
  return null;
};

const Simulation = () => {
  const [searchParams] = useSearchParams();
  const location = useLocation();
  const navigate = useNavigate();
  const initialAnomalyId = searchParams.get('anomaly_id') || location.state?.anomalyId || '';

  // Configuration State
  const [scenarioType, setScenarioType] = useState('demand_multiplier');
  const [horizonDays, setHorizonDays] = useState(30);
  const [demandChangePercent, setDemandChangePercent] = useState(10);
  const [shockDurationDays, setShockDurationDays] = useState(7);
  const [trendWindowDays, setTrendWindowDays] = useState(28);
  const [promotionActive, setPromotionActive] = useState(true);
  const [holidayActive, setHolidayActive] = useState(true);
  const [anomalyId, setAnomalyId] = useState(initialAnomalyId);
  const [includeRevenue, setIncludeRevenue] = useState(true);
  const [includeExplanation, setIncludeExplanation] = useState(true);

  // Execution State
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [simulationResult, setSimulationResult] = useState(null);
  const [submittingDecision, setSubmittingDecision] = useState(false);
  const [decisionSuccess, setDecisionSuccess] = useState(null);

  useEffect(() => {
    const incomingAnomalyId = searchParams.get('anomaly_id') || location.state?.anomalyId;
    if (incomingAnomalyId && incomingAnomalyId !== anomalyId) {
      setAnomalyId(incomingAnomalyId);
    }
  }, [searchParams, location.state]);

  // Auto-run when anchored from query param or on initial mount
  useEffect(() => {
    handleRunSimulation();
  }, []);

  const handleSendToDecisionReview = async () => {
    if (!simulationResult) return;
    try {
      setSubmittingDecision(true);
      setError(null);
      const deltaQty = simulationResult.summary?.quantity_delta ?? 0;
      const baseQty = simulationResult.summary?.baseline_quantity ?? 0;
      const scenQty = simulationResult.summary?.scenario_quantity ?? 0;
      const payload = {
        recommendation_type: 'forecast_review',
        proposed_action: `Evaluate hypothetical ${simulationResult.scenario_type.replace('_', ' ')} scenario over ${simulationResult.horizon_days} days (Projected volume variance: ${formatNumber(deltaQty)} units).`,
        anomaly_id: simulationResult.anomaly_id || null,
        simulation_id: simulationResult.simulation_id || null,
        decision_note: `What-if simulation submitted for executive review. Baseline: ${formatNumber(baseQty)} units, Scenario: ${formatNumber(scenQty)} units.`,
      };
      const created = await createDecisionApi(payload);
      setDecisionSuccess(`Successfully submitted to Decision Center (ID: ${created.id}). Redirecting...`);
      setTimeout(() => {
        navigate(`/decisions?id=${created.id}`);
      }, 700);
    } catch (err) {
      console.error('Failed to submit simulation to Decision Center:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to submit simulation to Decision Center.');
    } finally {
      setSubmittingDecision(false);
    }
  };

  const handleRunSimulation = async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = {
        scenario_type: scenarioType,
        horizon_days: horizonDays,
        demand_change_percent: ['demand_multiplier', 'temporary_shock', 'persistent_shift'].includes(scenarioType)
          ? demandChangePercent
          : null,
        shock_duration_days: scenarioType === 'temporary_shock' ? shockDurationDays : null,
        trend_window_days: scenarioType === 'trend_continuation' ? trendWindowDays : 28,
        promotion_active: promotionActive,
        holiday_active: holidayActive,
        anomaly_id: anomalyId.trim() ? anomalyId.trim() : null,
        include_revenue: includeRevenue,
        include_explanation: includeExplanation,
      };

      const result = await runSimulationApi(payload);
      setSimulationResult(result);
    } catch (err) {
      console.error('Simulation execution failed:', err);
      setError(
        err.response?.data?.detail || err.message || 'Failed to execute scenario simulation.'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center space-x-2 text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-1">
            <span>Financial Intelligence</span>
            <span>•</span>
            <span className="text-slate-400">Phase 6.6</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-100 flex items-center space-x-3">
            <span>What-If Scenario Simulation</span>
            <span className="text-xs px-2.5 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 font-mono font-medium">
              ML Guided
            </span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Simulate hypothetical demand shifts, market shocks, trends, and event lift without modifying historical sales records.
          </p>
        </div>

        {simulationResult && (
          <div className="flex items-center space-x-3">
            <div className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs">
              <span className="text-slate-400">Model: </span>
              <span className="font-semibold text-slate-200">{simulationResult.model_name}</span>
            </div>
            <div className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs">
              <span className="text-slate-400">Horizon: </span>
              <span className="font-mono font-semibold text-indigo-300">
                {simulationResult.horizon_days} Days
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Error Banner */}
      {error && (
        <div className="flex items-center space-x-3 p-4 rounded-xl bg-rose-950/40 border border-rose-800/50 text-rose-200 text-xs">
          <AlertTriangle className="w-5 h-5 text-rose-400 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Configuration Panel */}
      <SimulationPanel
        scenarioType={scenarioType}
        setScenarioType={setScenarioType}
        horizonDays={horizonDays}
        setHorizonDays={setHorizonDays}
        demandChangePercent={demandChangePercent}
        setDemandChangePercent={setDemandChangePercent}
        shockDurationDays={shockDurationDays}
        setShockDurationDays={setShockDurationDays}
        trendWindowDays={trendWindowDays}
        setTrendWindowDays={setTrendWindowDays}
        promotionActive={promotionActive}
        setPromotionActive={setPromotionActive}
        holidayActive={holidayActive}
        setHolidayActive={setHolidayActive}
        anomalyId={anomalyId}
        setAnomalyId={setAnomalyId}
        includeRevenue={includeRevenue}
        setIncludeRevenue={setIncludeRevenue}
        includeExplanation={includeExplanation}
        setIncludeExplanation={setIncludeExplanation}
        onRunSimulation={handleRunSimulation}
        loading={loading}
      />

      {/* Results View */}
      {simulationResult && (
        <div className="space-y-6">
          {/* Unsupported Model Notification */}
          {simulationResult.status === 'requires_model' && (
            <div className="bg-amber-950/20 border border-amber-800/40 rounded-2xl p-6 shadow-xl space-y-4">
              <div className="flex items-start space-x-3">
                <div className="p-2.5 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400">
                  <AlertTriangle className="w-6 h-6" />
                </div>
                <div className="space-y-1">
                  <h3 className="text-base font-bold text-amber-300">
                    Model Requirement: {simulationResult.required_model}
                  </h3>
                  <p className="text-xs text-amber-200/90 leading-relaxed">
                    {simulationResult.reason}
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
                <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4 space-y-2">
                  <span className="text-xs font-bold text-slate-300 uppercase tracking-wider block">
                    Modeling Boundary
                  </span>
                  <ul className="space-y-1 text-xs text-slate-400 list-disc list-inside">
                    {simulationResult.assumptions.map((a, idx) => (
                      <li key={idx}>{a}</li>
                    ))}
                  </ul>
                </div>
                <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-4 space-y-2">
                  <span className="text-xs font-bold text-slate-300 uppercase tracking-wider block">
                    Causal Safeguard
                  </span>
                  <ul className="space-y-1 text-xs text-slate-400 list-disc list-inside">
                    {simulationResult.limitations.map((l, idx) => (
                      <li key={idx}>{l}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}

          {/* Completed Simulation View */}
          {simulationResult.status === 'completed' && (
            <>
              {/* KPI Cards Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {/* 1. Baseline Total Quantity */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-2">
                  <div className="flex items-center justify-between text-xs text-slate-400 font-semibold uppercase tracking-wider">
                    <span>Baseline Forecast</span>
                    <span className="p-1.5 rounded-lg bg-indigo-500/10 text-indigo-400">
                      <TrendingUp className="w-4 h-4" />
                    </span>
                  </div>
                  <div className="text-2xl font-bold font-mono text-slate-100">
                    {formatQuantity(simulationResult.baseline?.total_quantity)}
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1 border-t border-slate-800/80">
                    <span>Daily Avg:</span>
                    <span className="font-mono text-slate-200">
                      {formatNumber(simulationResult.baseline?.average_daily_quantity)} units/day
                    </span>
                  </div>
                </div>

                {/* 2. Scenario Total Quantity */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-2">
                  <div className="flex items-center justify-between text-xs text-slate-400 font-semibold uppercase tracking-wider">
                    <span>Scenario Forecast</span>
                    <span className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400">
                      <Sparkles className="w-4 h-4" />
                    </span>
                  </div>
                  <div className="text-2xl font-bold font-mono text-emerald-400">
                    {formatQuantity(simulationResult.scenario?.total_quantity)}
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1 border-t border-slate-800/80">
                    <span>Daily Avg:</span>
                    <span className="font-mono text-slate-200">
                      {formatNumber(simulationResult.scenario?.average_daily_quantity)} units/day
                    </span>
                  </div>
                </div>

                {/* 3. Quantity Delta */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-2">
                  <div className="flex items-center justify-between text-xs text-slate-400 font-semibold uppercase tracking-wider">
                    <span>Volume Variance</span>
                    <span
                      className={`p-1.5 rounded-lg ${
                        (simulationResult.delta?.quantity_delta || 0) >= 0
                          ? 'bg-emerald-500/10 text-emerald-400'
                          : 'bg-rose-500/10 text-rose-400'
                      }`}
                    >
                      {(simulationResult.delta?.quantity_delta || 0) >= 0 ? (
                        <ArrowUpRight className="w-4 h-4" />
                      ) : (
                        <ArrowDownRight className="w-4 h-4" />
                      )}
                    </span>
                  </div>
                  <div
                    className={`text-2xl font-bold font-mono ${
                      (simulationResult.delta?.quantity_delta || 0) >= 0
                        ? 'text-emerald-400'
                        : 'text-rose-400'
                    }`}
                  >
                    {(simulationResult.delta?.quantity_delta || 0) >= 0 ? '+' : ''}
                    {formatNumber(simulationResult.delta?.quantity_delta)} units
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1 border-t border-slate-800/80">
                    <span>Percentage Shift:</span>
                    <span
                      className={`font-mono font-bold ${
                        (simulationResult.delta?.quantity_delta_percent || 0) >= 0
                          ? 'text-emerald-400'
                          : 'text-rose-400'
                      }`}
                    >
                      {(simulationResult.delta?.quantity_delta_percent || 0) >= 0 ? '+' : ''}
                      {formatPercent(simulationResult.delta?.quantity_delta_percent)}
                    </span>
                  </div>
                </div>

                {/* 4. Revenue Delta / Total */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-2">
                  <div className="flex items-center justify-between text-xs text-slate-400 font-semibold uppercase tracking-wider">
                    <span>Revenue Variance (₹)</span>
                    <span className="p-1.5 rounded-lg bg-sky-500/10 text-sky-400">
                      <IndianRupee className="w-4 h-4" />
                    </span>
                  </div>
                  <div
                    className={`text-2xl font-bold font-mono ${
                      (simulationResult.delta?.revenue_delta || 0) >= 0
                        ? 'text-sky-400'
                        : 'text-rose-400'
                    }`}
                  >
                    {simulationResult.delta?.revenue_delta !== null ? (
                      <>
                        {(simulationResult.delta?.revenue_delta || 0) >= 0 ? '+' : ''}
                        {formatCurrency(simulationResult.delta?.revenue_delta)}
                      </>
                    ) : (
                      'N/A'
                    )}
                  </div>
                  <div className="flex items-center justify-between text-[11px] text-slate-400 pt-1 border-t border-slate-800/80">
                    <span>Scenario Total:</span>
                    <span className="font-mono text-slate-200">
                      {simulationResult.scenario?.total_revenue !== null
                        ? formatCurrency(simulationResult.scenario?.total_revenue)
                        : 'Excluded'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Forecast Comparison Chart */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-slate-800/80 pb-3">
                  <div>
                    <h3 className="text-sm font-bold text-slate-100 flex items-center space-x-2">
                      <Activity className="w-4 h-4 text-indigo-400" />
                      <span>Forecast Trajectory: Baseline vs. Simulated Scenario</span>
                    </h3>
                    <p className="text-xs text-slate-400">
                      Daily units projected over {simulationResult.horizon_days} calendar days
                    </p>
                  </div>

                  <div className="flex items-center space-x-4 text-xs">
                    <div className="flex items-center space-x-1.5">
                      <span className="w-3 h-0.5 bg-indigo-500 rounded" />
                      <span className="text-slate-300">Baseline Forecast</span>
                    </div>
                    <div className="flex items-center space-x-1.5">
                      <span className="w-3 h-0.5 bg-emerald-400 rounded" />
                      <span className="text-slate-300">Simulated Scenario</span>
                    </div>
                  </div>
                </div>

                <div className="h-80 w-full pt-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={simulationResult.daily_results} margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                      <XAxis
                        dataKey="date"
                        tickFormatter={formatDate}
                        stroke="#64748b"
                        fontSize={11}
                        tickLine={false}
                        dy={8}
                      />
                      <YAxis
                        stroke="#64748b"
                        fontSize={11}
                        tickLine={false}
                        tickFormatter={(val) => `${(val / 1000).toFixed(1)}k`}
                        domain={['auto', 'auto']}
                      />
                      <Tooltip content={<CustomChartTooltip />} />
                      <Line
                        type="monotone"
                        dataKey="baseline_quantity"
                        name="Baseline"
                        stroke="#6366f1"
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 5, fill: '#6366f1' }}
                      />
                      <Line
                        type="monotone"
                        dataKey="scenario_quantity"
                        name="Scenario"
                        stroke="#10b981"
                        strokeWidth={2.5}
                        strokeDasharray="4 2"
                        dot={false}
                        activeDot={{ r: 5, fill: '#10b981' }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Narrative Explanation Card */}
              {simulationResult.explanation && (
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
                    <div className="flex items-center space-x-2.5">
                      <div className="p-2 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-400">
                        <BrainCircuit className="w-4 h-4" />
                      </div>
                      <h3 className="text-sm font-bold text-slate-100">Executive Narrative Summary</h3>
                    </div>
                    <span
                      className={`text-[10px] px-2.5 py-1 rounded-full font-semibold uppercase tracking-wider ${
                        simulationResult.source === 'gemini_explanation'
                          ? 'bg-purple-500/15 text-purple-300 border border-purple-500/30'
                          : 'bg-indigo-500/15 text-indigo-300 border border-indigo-500/30'
                      }`}
                    >
                      {simulationResult.source === 'gemini_explanation'
                        ? 'Gemini AI Explanation'
                        : 'Deterministic Rule Engine'}
                    </span>
                  </div>
                  <p className="text-xs sm:text-sm text-slate-200 leading-relaxed font-normal">
                    {simulationResult.explanation}
                  </p>
                </div>
              )}

              {/* Anomaly Context Card (if anchored) */}
              {simulationResult.anomaly_context && (
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800/80 pb-2.5">
                    <div className="flex items-center space-x-2 text-xs font-bold text-indigo-400 uppercase tracking-wider">
                      <Layers className="w-4 h-4" />
                      <span>Anchored Anomaly Investigation Context</span>
                    </div>
                    <span className="font-mono text-xs text-slate-400">
                      {simulationResult.anomaly_context.anomaly_id}
                    </span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 text-xs">
                    <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                      <span className="text-[10px] text-slate-400 block uppercase">Occurrence Date</span>
                      <span className="font-mono font-bold text-slate-200">
                        {formatDate(simulationResult.anomaly_context.anomaly_date)}
                      </span>
                    </div>
                    <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                      <span className="text-[10px] text-slate-400 block uppercase">Actual vs Expected</span>
                      <span className="font-mono font-bold text-slate-200">
                        {formatNumber(simulationResult.anomaly_context.actual_value)} vs{' '}
                        {formatNumber(simulationResult.anomaly_context.expected_baseline)}
                      </span>
                    </div>
                    <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                      <span className="text-[10px] text-slate-400 block uppercase">Historical Deviation</span>
                      <span className="font-mono font-bold text-rose-400">
                        {formatPercent(simulationResult.anomaly_context.deviation_percent)}
                      </span>
                    </div>
                    <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
                      <span className="text-[10px] text-slate-400 block uppercase">Severity</span>
                      <span className="font-semibold uppercase text-amber-400">
                        {simulationResult.anomaly_context.severity}
                      </span>
                    </div>
                  </div>
                  {simulationResult.anomaly_context.key_contributors?.length > 0 && (
                    <div className="pt-2">
                      <span className="text-[11px] font-semibold text-slate-400 block mb-1">
                        Key Historical Contributors:
                      </span>
                      <div className="flex flex-wrap gap-1.5">
                        {simulationResult.anomaly_context.key_contributors.map((c, i) => (
                          <span
                            key={i}
                            className="px-2.5 py-1 rounded-lg bg-slate-800 text-[11px] text-slate-300 border border-slate-700 font-mono"
                          >
                            {c}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Modeling Assumptions and Operational Limitations */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Assumptions */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
                  <div className="flex items-center space-x-2 text-xs font-bold text-slate-300 uppercase tracking-wider">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    <span>Explicit Modeling Assumptions</span>
                  </div>
                  <ul className="space-y-2 text-xs text-slate-300">
                    {simulationResult.assumptions.map((item, idx) => (
                      <li key={idx} className="flex items-start space-x-2 leading-relaxed">
                        <span className="text-emerald-400 font-bold mt-0.5">•</span>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Limitations */}
                <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
                  <div className="flex items-center space-x-2 text-xs font-bold text-slate-300 uppercase tracking-wider">
                    <ShieldAlert className="w-4 h-4 text-amber-400" />
                    <span>Methodological Limitations & Safeguards</span>
                  </div>
                  <ul className="space-y-2 text-xs text-slate-300">
                    {simulationResult.limitations.map((item, idx) => (
                      <li key={idx} className="flex items-start space-x-2 leading-relaxed">
                        <span className="text-amber-400 font-bold mt-0.5">•</span>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Decision Center Integration Card */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-5 rounded-2xl bg-gradient-to-r from-indigo-950/40 via-slate-900 to-slate-900 border border-indigo-500/30 shadow-xl">
                <div>
                  <div className="flex items-center space-x-2 text-xs font-bold text-indigo-400 uppercase tracking-wider">
                    <ShieldCheck className="w-4 h-4" />
                    <span>Executive Decision Center Integration</span>
                  </div>
                  <h4 className="text-sm font-bold text-slate-100 mt-0.5">
                    Promote Scenario to Governance Review Queue
                  </h4>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Anchor this hypothetical simulation to a formal decision record awaiting human review and sign-off.
                  </p>
                  {decisionSuccess && (
                    <p className="text-xs font-semibold text-emerald-400 mt-2 flex items-center space-x-1">
                      <CheckCircle2 className="w-3.5 h-3.5 mr-1" />
                      {decisionSuccess}
                    </p>
                  )}
                </div>
                <button
                  type="button"
                  onClick={handleSendToDecisionReview}
                  disabled={submittingDecision}
                  className="inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold shadow-lg shadow-indigo-950/50 transition-all cursor-pointer disabled:opacity-50 self-start sm:self-auto flex-shrink-0"
                >
                  <ShieldCheck className="w-4 h-4 text-indigo-200" />
                  <span>{submittingDecision ? 'Submitting...' : 'Send to Decision Center'}</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Human Approval Required Banner */}
              <div className="p-4 rounded-2xl bg-indigo-950/30 border border-indigo-800/40 text-xs text-indigo-200 flex items-start space-x-3 shadow-lg">
                <Info className="w-5 h-5 text-indigo-400 flex-shrink-0 mt-0.5" />
                <div className="space-y-1">
                  <span className="font-bold text-indigo-300">
                    Mandatory Human Approval Protocol (Phase 6.6)
                  </span>
                  <p className="text-indigo-200/90 leading-relaxed">
                    This what-if simulation is an exploratory analytical projection. It does not place purchase orders,
                    alter product pricing, update inventory allocations, or commit financial resources.
                    All operational decisions require qualified management verification and sign-off.
                  </p>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default Simulation;
