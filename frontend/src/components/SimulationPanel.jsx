import React from 'react';
import {
  Sliders,
  TrendingUp,
  TrendingDown,
  AlertCircle,
  AlertTriangle,
  Calendar,
  Sparkles,
  ShieldCheck,
  Info,
  Layers,
  ArrowRight,
} from 'lucide-react';

export const SCENARIO_OPTIONS = [
  {
    id: 'demand_multiplier',
    name: 'Demand Multiplier',
    description: 'Apply a uniform percentage shift across the entire forecast horizon.',
    badge: 'Standard',
  },
  {
    id: 'temporary_shock',
    name: 'Temporary Shock',
    description: 'Simulate a short-term spike or drop that reverts back to baseline.',
    badge: 'Reversion',
  },
  {
    id: 'persistent_shift',
    name: 'Persistent Shift',
    description: 'Evaluate a sustained structural shift in ongoing customer demand.',
    badge: 'Structural',
  },
  {
    id: 'trend_continuation',
    name: 'Trend Continuation',
    description: 'Project recent linear velocity from historical sales observations.',
    badge: 'Extrapolation',
  },
  {
    id: 'promotion_scenario',
    name: 'Promotional Campaign',
    description: 'Simulate calendar promotional lift using the production ML model.',
    badge: 'ML Feature',
  },
  {
    id: 'holiday_scenario',
    name: 'Holiday Trading',
    description: 'Simulate festive calendar lift using the production ML model.',
    badge: 'ML Feature',
  },
  {
    id: 'price_change',
    name: 'Price Change',
    description: 'Requires econometric price-elasticity model (Not supported by volume model).',
    badge: 'Requires Model',
    requiresModel: true,
  },
  {
    id: 'discount_change',
    name: 'Discount Depth',
    description: 'Requires discount elasticity model (Not supported by volume model).',
    badge: 'Requires Model',
    requiresModel: true,
  },
];

const SimulationPanel = ({
  scenarioType,
  setScenarioType,
  horizonDays,
  setHorizonDays,
  demandChangePercent,
  setDemandChangePercent,
  shockDurationDays,
  setShockDurationDays,
  trendWindowDays,
  setTrendWindowDays,
  promotionActive,
  setPromotionActive,
  holidayActive,
  setHolidayActive,
  anomalyId,
  setAnomalyId,
  includeRevenue,
  setIncludeRevenue,
  includeExplanation,
  setIncludeExplanation,
  onRunSimulation,
  loading,
}) => {
  const activeScenario = SCENARIO_OPTIONS.find((s) => s.id === scenarioType) || SCENARIO_OPTIONS[0];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
            <Sliders className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-base font-bold text-slate-100">Scenario Configuration</h2>
            <p className="text-xs text-slate-400">Select hypothetical assumptions and evaluate impact</p>
          </div>
        </div>
        <span className="text-[10px] font-semibold uppercase tracking-wider px-2.5 py-1 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
          Non-Destructive Simulation
        </span>
      </div>

      {/* Scenario Type Selection */}
      <div className="space-y-2">
        <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider block">
          Scenario Type
        </label>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
          {SCENARIO_OPTIONS.map((opt) => {
            const isSelected = scenarioType === opt.id;
            return (
              <button
                key={opt.id}
                type="button"
                onClick={() => setScenarioType(opt.id)}
                className={`text-left p-3 rounded-xl border transition-all cursor-pointer relative ${
                  isSelected
                    ? 'bg-indigo-600/15 border-indigo-500 text-slate-100 shadow-md shadow-indigo-950/40'
                    : opt.requiresModel
                    ? 'bg-slate-950/40 border-slate-800/80 text-slate-400 hover:border-slate-700 hover:bg-slate-800/30'
                    : 'bg-slate-950/60 border-slate-800 text-slate-300 hover:border-slate-700 hover:bg-slate-800/40'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs font-bold leading-snug">{opt.name}</span>
                  <span
                    className={`text-[9px] px-1.5 py-0.5 rounded font-mono font-medium ${
                      opt.requiresModel
                        ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                        : isSelected
                        ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30'
                        : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    {opt.badge}
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">
                  {opt.description}
                </p>
              </button>
            );
          })}
        </div>
      </div>

      {/* Scenario Parameters Form */}
      <div className="bg-slate-950/50 border border-slate-800/80 rounded-xl p-4 space-y-4">
        {activeScenario.requiresModel ? (
          <div className="flex items-start space-x-3.5 p-4 rounded-xl bg-amber-950/40 border border-amber-500/40 text-amber-200 shadow-lg">
            <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
            <div className="space-y-1.5 text-xs">
              <span className="font-bold text-sm text-amber-300 block">
                Model Boundary Restriction: Dedicated Elasticity Model Required
              </span>
              <p className="text-amber-200/90 leading-relaxed font-medium">
                This scenario requires a dedicated elasticity model and is not supported by the current forecasting model.
              </p>
              <p className="text-slate-300 text-[11px] leading-relaxed">
                The current production forecasting architecture forecasts volume from autoregressive and calendar features without explicit price sensitivity or discount depth parameters.
                Execution is disabled to avoid ungrounded commercial projections.
              </p>
            </div>
          </div>
        ) : null}

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Horizon Selection */}
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-slate-300 flex items-center space-x-1.5">
              <Calendar className="w-3.5 h-3.5 text-indigo-400" />
              <span>Forecast Horizon</span>
            </label>
            <div className="grid grid-cols-3 gap-1.5">
              {[7, 30, 90].map((h) => (
                <button
                  key={h}
                  type="button"
                  onClick={() => setHorizonDays(h)}
                  className={`py-2 text-xs font-semibold rounded-lg border transition-all cursor-pointer ${
                    horizonDays === h
                      ? 'bg-indigo-600 border-indigo-500 text-white shadow-sm'
                      : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                  }`}
                >
                  {h} Days
                </button>
              ))}
            </div>
          </div>

          {/* Conditional Input: Demand Multiplier / Shock / Shift */}
          {['demand_multiplier', 'temporary_shock', 'persistent_shift'].includes(scenarioType) && (
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-medium text-slate-300">Demand Shift</span>
                <span
                  className={`font-mono font-bold ${
                    demandChangePercent > 0
                      ? 'text-emerald-400'
                      : demandChangePercent < 0
                      ? 'text-rose-400'
                      : 'text-slate-300'
                  }`}
                >
                  {demandChangePercent > 0 ? `+${demandChangePercent}%` : `${demandChangePercent}%`}
                </span>
              </div>
              <input
                type="range"
                min="-50"
                max="50"
                step="1"
                value={demandChangePercent}
                onChange={(e) => setDemandChangePercent(Number(e.target.value))}
                className="w-full accent-indigo-500 cursor-pointer h-2 bg-slate-800 rounded-lg appearance-none"
              />
              <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                <span>-50%</span>
                <span>0%</span>
                <span>+50%</span>
              </div>
            </div>
          )}

          {/* Conditional Input: Shock Duration */}
          {scenarioType === 'temporary_shock' && (
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-medium text-slate-300">Shock Duration</span>
                <span className="font-mono font-bold text-indigo-400">{shockDurationDays} Days</span>
              </div>
              <input
                type="range"
                min="1"
                max={horizonDays}
                step="1"
                value={shockDurationDays}
                onChange={(e) => setShockDurationDays(Number(e.target.value))}
                className="w-full accent-indigo-500 cursor-pointer h-2 bg-slate-800 rounded-lg appearance-none"
              />
              <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                <span>1 Day</span>
                <span>{horizonDays} Days</span>
              </div>
            </div>
          )}

          {/* Conditional Input: Trend Window */}
          {scenarioType === 'trend_continuation' && (
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <span className="font-medium text-slate-300">Trend Evaluation Window</span>
                <span className="font-mono font-bold text-indigo-400">{trendWindowDays} Days</span>
              </div>
              <input
                type="range"
                min="7"
                max="90"
                step="1"
                value={trendWindowDays}
                onChange={(e) => setTrendWindowDays(Number(e.target.value))}
                className="w-full accent-indigo-500 cursor-pointer h-2 bg-slate-800 rounded-lg appearance-none"
              />
              <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                <span>7 Days</span>
                <span>28 Days</span>
                <span>90 Days</span>
              </div>
            </div>
          )}

          {/* Conditional Input: Promotional Scenario */}
          {scenarioType === 'promotion_scenario' && (
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-slate-300 block">Promotional State</label>
              <div className="grid grid-cols-2 gap-1.5">
                <button
                  type="button"
                  onClick={() => setPromotionActive(true)}
                  className={`py-2 text-xs font-semibold rounded-lg border transition-all cursor-pointer ${
                    promotionActive
                      ? 'bg-emerald-600/20 border-emerald-500 text-emerald-300'
                      : 'bg-slate-900 border-slate-800 text-slate-400'
                  }`}
                >
                  Active (1.0)
                </button>
                <button
                  type="button"
                  onClick={() => setPromotionActive(false)}
                  className={`py-2 text-xs font-semibold rounded-lg border transition-all cursor-pointer ${
                    !promotionActive
                      ? 'bg-slate-800 border-slate-700 text-slate-200'
                      : 'bg-slate-900 border-slate-800 text-slate-400'
                  }`}
                >
                  Inactive (0.0)
                </button>
              </div>
            </div>
          )}

          {/* Conditional Input: Holiday Scenario */}
          {scenarioType === 'holiday_scenario' && (
            <div className="space-y-1.5">
              <label className="text-xs font-medium text-slate-300 block">Holiday Trading State</label>
              <div className="grid grid-cols-2 gap-1.5">
                <button
                  type="button"
                  onClick={() => setHolidayActive(true)}
                  className={`py-2 text-xs font-semibold rounded-lg border transition-all cursor-pointer ${
                    holidayActive
                      ? 'bg-emerald-600/20 border-emerald-500 text-emerald-300'
                      : 'bg-slate-900 border-slate-800 text-slate-400'
                  }`}
                >
                  Active (1.0)
                </button>
                <button
                  type="button"
                  onClick={() => setHolidayActive(false)}
                  className={`py-2 text-xs font-semibold rounded-lg border transition-all cursor-pointer ${
                    !holidayActive
                      ? 'bg-slate-800 border-slate-700 text-slate-200'
                      : 'bg-slate-900 border-slate-800 text-slate-400'
                  }`}
                >
                  Inactive (0.0)
                </button>
              </div>
            </div>
          )}

          {/* Anomaly Anchor */}
          <div className="space-y-1.5">
            <label className="text-xs font-medium text-slate-300 flex items-center space-x-1.5">
              <Layers className="w-3.5 h-3.5 text-indigo-400" />
              <span>Anchor to Anomaly (Optional)</span>
            </label>
            <input
              type="text"
              placeholder="e.g. anom-20251228-sal-agg"
              value={anomalyId}
              onChange={(e) => setAnomalyId(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
            />
          </div>
        </div>

        {/* Toggles */}
        <div className="flex flex-wrap items-center gap-4 pt-2 border-t border-slate-800/80 text-xs">
          <label className="flex items-center space-x-2 cursor-pointer text-slate-300">
            <input
              type="checkbox"
              checked={includeRevenue}
              onChange={(e) => setIncludeRevenue(e.target.checked)}
              className="accent-indigo-500 rounded cursor-pointer"
            />
            <span>Include ₹ Revenue Simulation (Constant Realized Price)</span>
          </label>
          <label className="flex items-center space-x-2 cursor-pointer text-slate-300">
            <input
              type="checkbox"
              checked={includeExplanation}
              onChange={(e) => setIncludeExplanation(e.target.checked)}
              className="accent-indigo-500 rounded cursor-pointer"
            />
            <span>Generate Executive Narrative Explanation</span>
          </label>
        </div>
      </div>

      {/* Action CTA */}
      <div className="flex items-center justify-between pt-2">
        <div className="flex items-center space-x-2 text-[11px] text-slate-400">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <span>Strict zero-write simulation: historical sales records remain untouched.</span>
        </div>
        <button
          type="button"
          onClick={onRunSimulation}
          disabled={loading || activeScenario.requiresModel}
          title={
            activeScenario.requiresModel
              ? "This scenario requires a dedicated elasticity model and is not supported by the current forecasting model."
              : undefined
          }
          className={`inline-flex items-center space-x-2 px-5 py-2.5 rounded-xl text-xs font-bold shadow-lg transition-all ${
            activeScenario.requiresModel
              ? 'bg-slate-800 text-slate-500 border border-slate-700/80 cursor-not-allowed opacity-60'
              : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-950/50 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed'
          }`}
        >
          {loading ? (
            <>
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Computing Scenario...</span>
            </>
          ) : activeScenario.requiresModel ? (
            <>
              <AlertTriangle className="w-4 h-4 text-amber-500" />
              <span>Simulation Restricted</span>
            </>
          ) : (
            <>
              <span>Execute Simulation</span>
              <ArrowRight className="w-4 h-4" />
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default SimulationPanel;
