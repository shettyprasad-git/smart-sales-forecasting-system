import React, { useState, useEffect } from 'react';
import { getInvestigationApi } from '../api/investigations';
import { getExplanationApi } from '../api/explanations';
import { getAIReasoningApi } from '../api/aiReasoning';
import { extractErrorMessage } from '../api/axios';
import LoadingSpinner from './LoadingSpinner';
import ErrorMessage from './ErrorMessage';
import {
  X,
  ArrowUpRight,
  ArrowDownRight,
  Layers,
  Package,
  Tag,
  Calendar,
  IndianRupee,
  Activity,
  Sparkles,
  HelpCircle,
  TrendingUp,
  Percent,
  FileText,
  Copy,
  Check,
  ShieldCheck,
  BrainCircuit,
  RotateCw,
  AlertTriangle,
  Lightbulb,
} from 'lucide-react';
import {
  formatCurrency,
  formatQuantity,
  formatDate,
} from '../utils/formatters';

const SEVERITY_BADGES = {
  critical: 'bg-rose-500/10 border-rose-500/30 text-rose-400',
  high: 'bg-amber-500/10 border-amber-500/30 text-amber-400',
  medium: 'bg-yellow-500/10 border-yellow-500/30 text-yellow-300',
  low: 'bg-sky-500/10 border-sky-500/30 text-sky-400',
};

const CONFIDENCE_BADGES = {
  high: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400',
  medium: 'bg-amber-500/10 border-amber-500/20 text-amber-400',
  low: 'bg-slate-800 border-slate-700 text-slate-400',
};

const getDriverIcon = (driverType) => {
  switch (driverType) {
    case 'promotion':
      return Tag;
    case 'holiday':
      return Calendar;
    case 'category':
      return Layers;
    case 'product':
      return Package;
    case 'price':
      return IndianRupee;
    case 'discount':
      return Percent;
    case 'recent_trend':
    case 'baseline_drift':
      return TrendingUp;
    default:
      return Activity;
  }
};

const InvestigationDrawer = ({ anomalyId, isOpen, onClose }) => {
  const [activeTab, setActiveTab] = useState('investigation'); // 'investigation' | 'explanation' | 'ai_reasoning'
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);

  // Explanation state
  const [explLoading, setExplLoading] = useState(false);
  const [explError, setExplError] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [copied, setCopied] = useState(false);

  // AI Reasoning state
  const [aiLoading, setAiLoading] = useState(false);
  const [aiRefreshing, setAiRefreshing] = useState(false);
  const [aiError, setAiError] = useState(null);
  const [aiData, setAiData] = useState(null);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Reset tabs and cached state when anomalyId changes
  useEffect(() => {
    setActiveTab('investigation');
    setExplanation(null);
    setExplError(null);
    setAiData(null);
    setAiError(null);
  }, [anomalyId]);

  // Fetch Investigation data
  useEffect(() => {
    if (!isOpen || !anomalyId) return;

    const fetchInvestigation = async () => {
      try {
        setLoading(true);
        setError(null);
        const result = await getInvestigationApi(anomalyId);
        setData(result);
      } catch (err) {
        setError(extractErrorMessage(err));
      } finally {
        setLoading(false);
      }
    };

    fetchInvestigation();
  }, [isOpen, anomalyId]);

  // Fetch Explanation data when switching to explanation tab
  useEffect(() => {
    if (!isOpen || !anomalyId || activeTab !== 'explanation') return;
    if (explanation) return; // Already loaded

    const fetchExplanation = async () => {
      try {
        setExplLoading(true);
        setExplError(null);
        const result = await getExplanationApi(anomalyId);
        setExplanation(result);
      } catch (err) {
        setExplError(extractErrorMessage(err));
      } finally {
        setExplLoading(false);
      }
    };

    fetchExplanation();
  }, [isOpen, anomalyId, activeTab, explanation]);

  // Fetch AI Reasoning data when switching to ai_reasoning tab
  useEffect(() => {
    if (!isOpen || !anomalyId || activeTab !== 'ai_reasoning') return;
    if (aiData) return; // Already loaded

    const fetchReasoning = async () => {
      try {
        setAiLoading(true);
        setAiError(null);
        const result = await getAIReasoningApi(anomalyId, { refresh: false });
        setAiData(result);
      } catch (err) {
        setAiError(extractErrorMessage(err));
      } finally {
        setAiLoading(false);
      }
    };

    fetchReasoning();
  }, [isOpen, anomalyId, activeTab, aiData]);

  const handleRefreshAI = async () => {
    if (!anomalyId) return;
    try {
      setAiRefreshing(true);
      setAiError(null);
      const result = await getAIReasoningApi(anomalyId, { refresh: true });
      setAiData(result);
    } catch (err) {
      setAiError(extractErrorMessage(err));
    } finally {
      setAiRefreshing(false);
    }
  };

  const handleCopyBrief = () => {
    if (!explanation) return;

    const contributorsList = (explanation.key_contributors || [])
      .map((c) => `- ${c}`)
      .join('\n');

    const limitationsList = (explanation.limitations || [])
      .map((l) => `- ${l}`)
      .join('\n');

    const markdownText = `# EXECUTIVE SALES ANOMALY BRIEF
**Anomaly ID:** ${explanation.anomaly_id}

## Headline
${explanation.headline}

## What Happened
${explanation.what_happened}

## Why It Matters
${explanation.why_it_matters}

## Key Contributors
${contributorsList || '- No dominant primary contributor.'}

## Evidence Quality & Confidence
${explanation.confidence_summary}

## Methodological Limitations
${limitationsList}
`;

    navigator.clipboard.writeText(markdownText).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  if (!isOpen) return null;

  const isSpike = data?.direction === 'spike';
  const isQuantity = data?.metric === 'quantity';
  const impact = data?.estimated_impact;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div
        className="relative w-full max-w-4xl bg-slate-900 border border-slate-800 rounded-3xl shadow-2xl overflow-hidden flex flex-col max-h-[92vh]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/70">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h3 className="text-base font-bold text-slate-100">
                  Root-Cause Investigation
                </h3>
                {data && (
                  <span
                    className={`px-2 py-0.5 rounded-full text-[10px] font-semibold border ${
                      SEVERITY_BADGES[data.severity] || SEVERITY_BADGES.low
                    }`}
                  >
                    {data.severity.toUpperCase()}
                  </span>
                )}
              </div>
              <p className="text-[11px] text-slate-400 font-mono mt-0.5">
                {anomalyId}
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            {/* View Switcher Tabs in Header */}
            <div className="flex items-center bg-slate-950/60 p-1 rounded-xl border border-slate-800 text-xs font-semibold">
              <button
                onClick={() => setActiveTab('investigation')}
                className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer ${
                  activeTab === 'investigation'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                Drivers & Breakdown
              </button>
              <button
                onClick={() => setActiveTab('explanation')}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg transition-all cursor-pointer ${
                  activeTab === 'explanation'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <FileText className="w-3.5 h-3.5" />
                <span>Executive Brief</span>
              </button>
              <button
                onClick={() => setActiveTab('ai_reasoning')}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg transition-all cursor-pointer ${
                  activeTab === 'ai_reasoning'
                    ? 'bg-purple-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                <BrainCircuit className="w-3.5 h-3.5 text-purple-300" />
                <span>AI Reasoning</span>
              </button>
            </div>

            <button
              onClick={onClose}
              className="p-1.5 rounded-xl text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors cursor-pointer"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Content Body */}
        <div className="p-6 overflow-y-auto custom-scrollbar space-y-6 flex-1">
          {error ? (
            <ErrorMessage
              title="Investigation Failed"
              message={error}
              onRetry={() => {
                if (anomalyId) {
                  setLoading(true);
                  setError(null);
                  getInvestigationApi(anomalyId)
                    .then(setData)
                    .catch((err) => setError(extractErrorMessage(err)))
                    .finally(() => setLoading(false));
                }
              }}
            />
          ) : loading || !data ? (
            <div className="py-12">
              <LoadingSpinner
                text="Decomposing drivers across categories, products, events, and price baselines..."
                size="lg"
              />
            </div>
          ) : activeTab === 'investigation' ? (
            /* TAB 1: Drivers & Empirical Breakdown */
            <>
              {/* Anomaly Context & Impact Grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Observation Metrics */}
                <div className="rounded-2xl bg-slate-950/60 border border-slate-800 p-4 space-y-2">
                  <div className="text-[10px] font-bold text-slate-500 uppercase tracking-wider">
                    Observed Anomaly
                  </div>
                  <div className="text-lg font-bold text-slate-100 font-mono">
                    {isQuantity
                      ? formatQuantity(data.actual_value)
                      : formatCurrency(data.actual_value)}
                  </div>
                  <div className="text-xs text-slate-400">
                    Baseline:{' '}
                    <span className="font-mono text-slate-300">
                      {isQuantity
                        ? formatQuantity(data.expected_baseline)
                        : formatCurrency(data.expected_baseline)}
                    </span>
                  </div>
                  <div className="flex items-center space-x-1 text-xs pt-1">
                    {isSpike ? (
                      <ArrowUpRight className="w-3.5 h-3.5 text-emerald-400" />
                    ) : (
                      <ArrowDownRight className="w-3.5 h-3.5 text-rose-400" />
                    )}
                    <span
                      className={`font-mono font-bold ${
                        isSpike ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {isSpike ? `+${data.deviation_percent}%` : `${data.deviation_percent}%`}
                    </span>
                    <span className="text-[10px] text-slate-500">
                      (Score: {data.anomaly_score.toFixed(2)})
                    </span>
                  </div>
                </div>

                {/* Estimated Business Impact */}
                <div className="rounded-2xl bg-slate-950/60 border border-slate-800 p-4 space-y-2 md:col-span-2">
                  <div className="flex items-center justify-between text-[10px] font-bold text-slate-500 uppercase tracking-wider">
                    <span>Quantified Impact</span>
                    <span className="text-indigo-400 font-normal">
                      Date: {formatDate(data.anomaly_date)}
                    </span>
                  </div>
                  <div className="text-lg font-bold font-mono text-slate-100">
                    {impact?.estimated_revenue_impact != null
                      ? formatCurrency(impact.estimated_revenue_impact)
                      : `${formatQuantity(impact?.impact_value)} impact`}
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    {impact?.interpretation?.replace(/\$/g, '₹')}
                  </p>
                  {impact?.price_basis_explanation && (
                    <p className="text-[11px] text-slate-500 italic">
                      Note: {impact.price_basis_explanation.replace(/\$/g, '₹')}
                    </p>
                  )}
                </div>
              </div>

              {/* Executive Summary & Quick Action Buttons */}
              <div className="rounded-2xl bg-gradient-to-r from-indigo-950/40 via-slate-900 to-slate-900 border border-indigo-500/20 p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div className="space-y-1 flex-1">
                  <div className="flex items-center space-x-2 text-xs font-bold text-indigo-300">
                    <Sparkles className="w-4 h-4 text-indigo-400" />
                    <span>Deterministic Evidence Summary</span>
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    {data.investigation_summary?.replace(/\$/g, '₹')}
                  </p>
                </div>
                <div className="flex items-center space-x-2 shrink-0">
                  <button
                    onClick={() => setActiveTab('explanation')}
                    className="inline-flex items-center space-x-1.5 px-3 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/20 transition-all cursor-pointer"
                  >
                    <FileText className="w-3.5 h-3.5" />
                    <span>View Brief</span>
                  </button>
                  <button
                    onClick={() => setActiveTab('ai_reasoning')}
                    className="inline-flex items-center space-x-1.5 px-3 py-2 rounded-xl bg-purple-600 hover:bg-purple-500 text-white text-xs font-semibold shadow-lg shadow-purple-600/20 transition-all cursor-pointer"
                  >
                    <BrainCircuit className="w-3.5 h-3.5" />
                    <span>AI Reasoning</span>
                  </button>
                </div>
              </div>

              {/* Contributing Drivers Section */}
              <div className="space-y-3">
                <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                  <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center space-x-2">
                    <Layers className="w-4 h-4 text-indigo-400" />
                    <span>Ranked Associated Drivers</span>
                  </h4>
                  <span className="text-[11px] text-slate-500">
                    {data.drivers.length} factors evaluated
                  </span>
                </div>

                <div className="space-y-3">
                  {data.drivers.map((driver, idx) => {
                    const DriverIcon = getDriverIcon(driver.driver_type);
                    const confClass =
                      CONFIDENCE_BADGES[driver.confidence] || CONFIDENCE_BADGES.low;

                    return (
                      <div
                        key={idx}
                        className="rounded-2xl bg-slate-950/70 border border-slate-800/80 p-4 space-y-2 hover:border-slate-700/80 transition-colors"
                      >
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                          <div className="flex items-center space-x-2.5">
                            <div className="p-1.5 rounded-lg bg-slate-800 text-slate-300">
                              <DriverIcon className="w-4 h-4 text-indigo-400" />
                            </div>
                            <div>
                              <span className="text-xs font-bold text-slate-100">
                                {driver.driver_name}
                              </span>
                              <span className="text-[10px] text-slate-500 ml-2 uppercase">
                                {driver.driver_type.replace('_', ' ')}
                              </span>
                            </div>
                          </div>

                          <div className="flex items-center space-x-2">
                            {driver.contribution_score > 0 && (
                              <span className="px-2 py-0.5 rounded-md bg-indigo-950/60 border border-indigo-500/20 text-indigo-300 text-[10px] font-mono font-bold">
                                {driver.contribution_score}% contribution
                              </span>
                            )}
                            <span
                              className={`px-2 py-0.5 rounded-md text-[10px] font-semibold border ${confClass}`}
                            >
                              {driver.confidence.toUpperCase()} CONFIDENCE
                            </span>
                          </div>
                        </div>

                        {/* Relative Contribution Bar */}
                        {driver.contribution_score > 0 && (
                          <div className="w-full bg-slate-800/60 rounded-full h-1.5 overflow-hidden">
                            <div
                              className="bg-indigo-500 h-full rounded-full transition-all duration-300"
                              style={{ width: `${Math.min(100, driver.contribution_score)}%` }}
                            />
                          </div>
                        )}

                        <p className="text-xs text-slate-400 leading-relaxed">
                          {driver.evidence?.replace(/\$/g, '₹')}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Limitations & Observational Disclaimer */}
              <div className="rounded-2xl bg-slate-950/40 border border-slate-800/60 p-4 space-y-2 text-[11px] text-slate-500">
                <div className="flex items-center space-x-1.5 font-bold text-slate-400 uppercase tracking-wider text-[10px]">
                  <HelpCircle className="w-3.5 h-3.5 text-slate-400" />
                  <span>Observational Attribution & Methodology Constraints</span>
                </div>
                <ul className="list-disc list-inside space-y-1 pl-1">
                  {data.limitations.map((lim, idx) => (
                    <li key={idx}>{lim}</li>
                  ))}
                </ul>
              </div>
            </>
          ) : activeTab === 'explanation' ? (
            /* TAB 2: Phase 6.3 Executive Brief */
            <div className="space-y-6">
              {explError ? (
                <ErrorMessage
                  title="Executive Brief Generation Failed"
                  message={explError}
                  onRetry={() => {
                    setExplLoading(true);
                    setExplError(null);
                    getExplanationApi(anomalyId)
                      .then(setExplanation)
                      .catch((err) => setExplError(extractErrorMessage(err)))
                      .finally(() => setExplLoading(false));
                  }}
                />
              ) : explLoading || !explanation ? (
                <div className="py-12">
                  <LoadingSpinner
                    text="Generating deterministic executive brief and narrative sections..."
                    size="lg"
                  />
                </div>
              ) : (
                <>
                  {/* Executive Headline & Quick Action Bar */}
                  <div className="rounded-2xl bg-gradient-to-r from-indigo-950/60 via-slate-900 to-slate-900 border border-indigo-500/30 p-5 space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-indigo-500/20 border border-indigo-500/30 text-indigo-300">
                          Executive Narrative Brief
                        </span>
                        <span className="text-xs text-slate-400 font-mono">
                          {explanation.anomaly_id}
                        </span>
                      </div>

                      <div className="flex items-center space-x-2">
                        <button
                          onClick={() => setActiveTab('ai_reasoning')}
                          className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl bg-purple-600 hover:bg-purple-500 text-white text-xs font-semibold shadow-sm transition-colors cursor-pointer"
                        >
                          <BrainCircuit className="w-3.5 h-3.5" />
                          <span>View AI Reasoning</span>
                        </button>
                        <button
                          onClick={handleCopyBrief}
                          className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition-colors cursor-pointer"
                        >
                          {copied ? (
                            <>
                              <Check className="w-3.5 h-3.5 text-emerald-400" />
                              <span className="text-emerald-400">Copied Brief</span>
                            </>
                          ) : (
                            <>
                              <Copy className="w-3.5 h-3.5 text-slate-400" />
                              <span>Copy Brief</span>
                            </>
                          )}
                        </button>
                      </div>
                    </div>

                    <h2 className="text-lg sm:text-xl font-bold text-slate-100 leading-snug">
                      {explanation.headline}
                    </h2>
                  </div>

                  {/* 2-Column: What Happened & Why It Matters */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* What Happened Card */}
                    <div className="rounded-2xl bg-slate-950/70 border border-slate-800 p-4 space-y-2">
                      <div className="flex items-center space-x-2 text-xs font-bold text-slate-300 uppercase tracking-wider">
                        <Activity className="w-4 h-4 text-indigo-400" />
                        <span>What Happened</span>
                      </div>
                      <p className="text-xs text-slate-300 leading-relaxed">
                        {explanation.what_happened}
                      </p>
                    </div>

                    {/* Why It Matters Card */}
                    <div className="rounded-2xl bg-slate-950/70 border border-slate-800 p-4 space-y-2">
                      <div className="flex items-center space-x-2 text-xs font-bold text-slate-300 uppercase tracking-wider">
                        <IndianRupee className="w-4 h-4 text-emerald-400" />
                        <span>Why It Matters</span>
                      </div>
                      <p className="text-xs text-slate-300 leading-relaxed">
                        {explanation.why_it_matters}
                      </p>
                    </div>
                  </div>

                  {/* Prioritized Key Contributors */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                      <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center space-x-2">
                        <Layers className="w-4 h-4 text-indigo-400" />
                        <span>Prioritized Contributing Drivers (Top {explanation.key_contributors.length})</span>
                      </h4>
                      <span className="text-[11px] text-slate-500">
                        Ranked by contribution & directional relevance
                      </span>
                    </div>

                    <div className="space-y-2.5">
                      {explanation.key_contributors.map((bullet, idx) => (
                        <div
                          key={idx}
                          className="rounded-xl bg-slate-950/60 border border-slate-800/80 p-3.5 flex items-start space-x-3 text-xs text-slate-300 leading-relaxed"
                        >
                          <span className="w-5 h-5 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 font-mono text-[11px] font-bold flex items-center justify-center shrink-0 mt-0.5">
                            {idx + 1}
                          </span>
                          <span className="flex-1">{bullet}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Context Cards: Event Context & Trend Context */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Event Context */}
                    <div className="rounded-2xl bg-slate-950/60 border border-slate-800 p-4 space-y-2">
                      <div className="flex items-center space-x-2 text-xs font-bold text-slate-300 uppercase tracking-wider">
                        <Calendar className="w-4 h-4 text-amber-400" />
                        <span>Event & Calendar Context</span>
                      </div>
                      <p className="text-xs text-slate-400 leading-relaxed">
                        {explanation.sections.find((s) => s.section_type === 'event_context')?.content}
                      </p>
                    </div>

                    {/* Trend Context */}
                    <div className="rounded-2xl bg-slate-950/60 border border-slate-800 p-4 space-y-2">
                      <div className="flex items-center space-x-2 text-xs font-bold text-slate-300 uppercase tracking-wider">
                        <TrendingUp className="w-4 h-4 text-sky-400" />
                        <span>Trend & Drift Context</span>
                      </div>
                      <p className="text-xs text-slate-400 leading-relaxed">
                        {explanation.sections.find((s) => s.section_type === 'trend_context')?.content}
                      </p>
                    </div>
                  </div>

                  {/* Evidence Quality & Confidence */}
                  <div className="rounded-2xl bg-slate-950/70 border border-slate-800 p-4 space-y-2">
                    <div className="flex items-center space-x-2 text-xs font-bold text-slate-300 uppercase tracking-wider">
                      <ShieldCheck className="w-4 h-4 text-emerald-400" />
                      <span>Evidence Quality & Confidence Assessment</span>
                    </div>
                    <p className="text-xs text-slate-300 leading-relaxed">
                      {explanation.confidence_summary}
                    </p>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      {explanation.sections.find((s) => s.section_type === 'evidence_quality')?.content}
                    </p>
                  </div>

                  {/* Methodological Limitations */}
                  <div className="rounded-2xl bg-slate-950/40 border border-slate-800/60 p-4 space-y-2 text-[11px] text-slate-500">
                    <div className="flex items-center space-x-1.5 font-bold text-slate-400 uppercase tracking-wider text-[10px]">
                      <HelpCircle className="w-3.5 h-3.5 text-slate-400" />
                      <span>Observational Attribution & Methodology Limitations</span>
                    </div>
                    <ul className="list-disc list-inside space-y-1 pl-1">
                      {explanation.limitations.map((lim, idx) => (
                        <li key={idx}>{lim}</li>
                      ))}
                    </ul>
                  </div>
                </>
              )}
            </div>
          ) : (
            /* TAB 3: Phase 6.4 AI Reasoning Layer */
            <div className="space-y-6">
              {aiError ? (
                <div className="rounded-2xl bg-amber-950/30 border border-amber-500/30 p-5 space-y-3">
                  <div className="flex items-center space-x-2 text-amber-400 font-semibold text-sm">
                    <AlertTriangle className="w-5 h-5" />
                    <span>AI Reasoning Currently Unavailable</span>
                  </div>
                  <p className="text-xs text-slate-300 leading-relaxed">
                    {aiError}
                  </p>
                  <p className="text-[11px] text-slate-400">
                    Deterministic statistical anomaly detection, root-cause driver attribution, and executive briefs remain 100% operational.
                  </p>
                  <div className="pt-2">
                    <button
                      onClick={() => handleRefreshAI()}
                      className="px-3 py-1.5 rounded-lg bg-amber-500/20 border border-amber-500/40 text-amber-300 text-xs font-semibold hover:bg-amber-500/30 transition-colors cursor-pointer"
                    >
                      Retry Reasoning
                    </button>
                  </div>
                </div>
              ) : aiLoading || !aiData ? (
                <div className="py-12">
                  <LoadingSpinner
                    text="Synthesizing multi-dimensional evidence with Gemini AI reasoning layer..."
                    size="lg"
                  />
                </div>
              ) : (
                <>
                  {/* AI Reasoning Header & Actions */}
                  <div className="rounded-2xl bg-gradient-to-r from-purple-950/60 via-slate-900 to-slate-900 border border-purple-500/30 p-5 space-y-3">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                      <div className="flex items-center space-x-2">
                        <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-purple-500/20 border border-purple-500/30 text-purple-300 flex items-center space-x-1">
                          <BrainCircuit className="w-3 h-3" />
                          <span>AI-GENERATED REASONING</span>
                        </span>
                        <span className="text-[11px] text-slate-400 italic">
                          Grounded in statistical evidence
                        </span>
                      </div>

                      <div className="flex items-center space-x-2">
                        {aiData.cached && (
                          <span className="text-[10px] font-mono text-slate-500 uppercase px-2 py-0.5 rounded-md bg-slate-800 border border-slate-700">
                            Cached
                          </span>
                        )}
                        <button
                          onClick={handleRefreshAI}
                          disabled={aiRefreshing}
                          className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition-colors cursor-pointer disabled:opacity-50"
                        >
                          <RotateCw className={`w-3.5 h-3.5 text-purple-400 ${aiRefreshing ? 'animate-spin' : ''}`} />
                          <span>{aiRefreshing ? 'Refreshing...' : 'Refresh Analysis'}</span>
                        </button>
                      </div>
                    </div>

                    <h2 className="text-lg sm:text-xl font-bold text-slate-100 leading-snug">
                      {aiData.reasoning_headline}
                    </h2>
                  </div>

                  {/* Executive Interpretation */}
                  <div className="rounded-2xl bg-slate-950/70 border border-slate-800 p-5 space-y-2">
                    <div className="flex items-center space-x-2 text-xs font-bold text-purple-400 uppercase tracking-wider">
                      <Lightbulb className="w-4 h-4" />
                      <span>Executive Interpretation</span>
                    </div>
                    <p className="text-xs text-slate-300 leading-relaxed">
                      {aiData.executive_interpretation}
                    </p>
                  </div>

                  {/* Key Insights Grid */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                      <h4 className="text-xs font-bold text-slate-200 uppercase tracking-wider flex items-center space-x-2">
                        <Sparkles className="w-4 h-4 text-purple-400" />
                        <span>Key Evidence-Grounded Insights ({aiData.key_insights.length})</span>
                      </h4>
                      <span className="text-[11px] text-slate-500">
                        Synthesized from deterministic drivers
                      </span>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {aiData.key_insights.map((insight, idx) => {
                        const confClass =
                          CONFIDENCE_BADGES[insight.confidence] || CONFIDENCE_BADGES.low;

                        return (
                          <div
                            key={idx}
                            className="rounded-2xl bg-slate-950/60 border border-slate-800/80 p-4 space-y-2 flex flex-col justify-between"
                          >
                            <div className="space-y-2">
                              <div className="flex items-center justify-between gap-2">
                                <span className="text-xs font-bold text-slate-100 leading-snug">
                                  {insight.statement}
                                </span>
                                <span
                                  className={`px-2 py-0.5 rounded-md text-[10px] font-semibold border shrink-0 ${confClass}`}
                                >
                                  {insight.confidence.toUpperCase()}
                                </span>
                              </div>
                              <p className="text-xs text-slate-400 leading-relaxed">
                                {insight.supporting_evidence}
                              </p>
                            </div>
                            {insight.related_driver && (
                              <div className="pt-2">
                                <span className="text-[10px] px-2 py-0.5 rounded-md bg-slate-900 border border-slate-800 text-slate-400 font-mono">
                                  Driver: {insight.related_driver}
                                </span>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* 2-Column: Alternative Explanations & Evidence Assessment */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Alternative Explanations */}
                    <div className="rounded-2xl bg-slate-950/70 border border-slate-800 p-4 space-y-2">
                      <div className="flex items-center space-x-2 text-xs font-bold text-slate-300 uppercase tracking-wider">
                        <Layers className="w-4 h-4 text-indigo-400" />
                        <span>Alternative Explanations</span>
                      </div>
                      <p className="text-[11px] text-slate-500 italic">
                        Competing hypotheses supported by empirical patterns:
                      </p>
                      <ul className="space-y-1.5 text-xs text-slate-300">
                        {aiData.alternative_explanations.map((alt, idx) => (
                          <li key={idx} className="flex items-start space-x-2">
                            <span className="text-indigo-400 mt-0.5">•</span>
                            <span>{alt}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* Evidence Assessment */}
                    <div className="rounded-2xl bg-slate-950/70 border border-slate-800 p-4 space-y-2">
                      <div className="flex items-center space-x-2 text-xs font-bold text-slate-300 uppercase tracking-wider">
                        <ShieldCheck className="w-4 h-4 text-emerald-400" />
                        <span>Evidence Strength Assessment</span>
                      </div>
                      <p className="text-xs text-slate-300 leading-relaxed">
                        {aiData.evidence_assessment}
                      </p>
                    </div>
                  </div>

                  {/* 2-Column: Uncertainties & Validation Questions */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Uncertainties */}
                    <div className="rounded-2xl bg-slate-950/60 border border-slate-800 p-4 space-y-2">
                      <div className="flex items-center space-x-2 text-xs font-bold text-slate-300 uppercase tracking-wider">
                        <HelpCircle className="w-4 h-4 text-amber-400" />
                        <span>Identified Uncertainties</span>
                      </div>
                      <ul className="space-y-1.5 text-xs text-slate-400">
                        {aiData.uncertainties.map((unc, idx) => (
                          <li key={idx} className="flex items-start space-x-2">
                            <span className="text-amber-400 mt-0.5">•</span>
                            <span>{unc}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* Validation Questions */}
                    <div className="rounded-2xl bg-slate-950/60 border border-slate-800 p-4 space-y-2">
                      <div className="flex items-center space-x-2 text-xs font-bold text-slate-300 uppercase tracking-wider">
                        <Check className="w-4 h-4 text-sky-400" />
                        <span>Validation Questions for Stakeholders</span>
                      </div>
                      <ul className="space-y-1.5 text-xs text-slate-300">
                        {aiData.validation_questions.map((q, idx) => (
                          <li key={idx} className="flex items-start space-x-2">
                            <span className="text-sky-400 font-mono text-xs">{idx + 1}.</span>
                            <span>{q}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  {/* Risk Flags */}
                  {aiData.risk_flags && aiData.risk_flags.length > 0 && (
                    <div className="rounded-2xl bg-slate-950/70 border border-rose-950/50 p-4 space-y-2">
                      <div className="flex items-center space-x-2 text-xs font-bold text-rose-400 uppercase tracking-wider">
                        <AlertTriangle className="w-4 h-4" />
                        <span>Risk Flags & Critical Attention Areas</span>
                      </div>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
                        {aiData.risk_flags.map((risk, idx) => (
                          <div
                            key={idx}
                            className="rounded-xl bg-rose-950/20 border border-rose-500/20 p-2.5 text-xs text-rose-300 flex items-start space-x-2"
                          >
                            <span className="text-rose-400 font-bold">•</span>
                            <span>{risk}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Mandatory Causal Disclaimer Banner */}
                  <div className="rounded-2xl bg-slate-950/40 border border-slate-800/80 p-4 flex items-center space-x-3 text-xs text-slate-400">
                    <ShieldCheck className="w-5 h-5 text-indigo-400 shrink-0" />
                    <p className="leading-relaxed">
                      <strong className="text-slate-300">Scientific Attribution Disclaimer:</strong>{' '}
                      {aiData.causal_disclaimer}
                    </p>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default InvestigationDrawer;
