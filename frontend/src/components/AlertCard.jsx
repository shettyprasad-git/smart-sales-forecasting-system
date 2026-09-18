import React from 'react';
import {
  TrendingUp,
  TrendingDown,
  Clock,
  Calendar,
  Layers,
  Search,
  Sliders,
  ShieldCheck,
  CheckCircle2,
  XCircle,
  Eye,
  ChevronRight,
} from 'lucide-react';

const AlertCard = ({
  alert,
  onSelect,
  onAcknowledge,
  onResolve,
  onDismiss,
  onInvestigate,
  onSimulate,
  onReview,
  actionLoading,
}) => {
  const isSpike = alert.deviation >= 0;
  const isCurrency = alert.metric === 'sales_amount';

  const formatValue = (val) => {
    if (val === null || val === undefined) return '—';
    if (isCurrency) {
      return `₹${Number(val).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
    }
    return `${Number(val).toLocaleString()} units`;
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case 'new':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold bg-indigo-900/60 text-indigo-300 border border-indigo-700/60 uppercase tracking-wider">
            ● New
          </span>
        );
      case 'acknowledged':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-900/60 text-amber-300 border border-amber-700/60 uppercase tracking-wider">
            Acknowledged
          </span>
        );
      case 'resolved':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-900/60 text-emerald-300 border border-emerald-700/60 uppercase tracking-wider">
            Resolved
          </span>
        );
      case 'dismissed':
        return (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold bg-slate-800 text-slate-400 border border-slate-700 uppercase tracking-wider">
            Dismissed
          </span>
        );
      default:
        return null;
    }
  };

  const getSeverityBadge = (sev) => {
    switch (sev) {
      case 'critical':
        return (
          <span className="px-2 py-0.5 rounded-md text-[10px] font-bold bg-rose-950/70 text-rose-300 border border-rose-800/80">
            CRITICAL
          </span>
        );
      case 'high':
        return (
          <span className="px-2 py-0.5 rounded-md text-[10px] font-bold bg-amber-950/70 text-amber-300 border border-amber-800/80">
            HIGH
          </span>
        );
      case 'medium':
        return (
          <span className="px-2 py-0.5 rounded-md text-[10px] font-bold bg-blue-950/70 text-blue-300 border border-blue-800/80">
            MEDIUM
          </span>
        );
      default:
        return (
          <span className="px-2 py-0.5 rounded-md text-[10px] font-bold bg-slate-800 text-slate-400 border border-slate-700">
            LOW
          </span>
        );
    }
  };

  const getPriorityBadge = (prio) => {
    switch (prio) {
      case 'urgent':
        return <span className="text-[10px] font-bold text-rose-400">P1 • Urgent</span>;
      case 'high':
        return <span className="text-[10px] font-bold text-amber-400">P2 • High</span>;
      case 'medium':
        return <span className="text-[10px] font-bold text-sky-400">P3 • Medium</span>;
      default:
        return <span className="text-[10px] font-bold text-slate-400">P4 • Low</span>;
    }
  };

  return (
    <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 hover:border-slate-700 transition-all duration-200 shadow-lg group">
      {/* Header Row */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div className="flex items-center space-x-2">
          {getStatusBadge(alert.status)}
          {getSeverityBadge(alert.severity)}
          <span className="text-slate-600">•</span>
          {getPriorityBadge(alert.priority)}
        </div>
        <div className="flex items-center space-x-2 text-xs text-slate-400">
          <Calendar className="w-3.5 h-3.5 text-slate-500" />
          <span>{alert.event_date}</span>
        </div>
      </div>

      {/* Title */}
      <div className="mb-4">
        <h3
          onClick={() => onSelect?.(alert)}
          className="text-sm font-bold text-slate-100 group-hover:text-indigo-400 transition-colors cursor-pointer flex items-center space-x-1.5"
        >
          <span>{alert.title}</span>
          <ChevronRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity text-indigo-400" />
        </h3>
        <p className="text-xs text-slate-400 mt-1 line-clamp-1">
          {alert.evidence_snapshot?.explanation || `Empirical deviation observed in ${alert.metric.replace('_', ' ')}.`}
        </p>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-3 gap-3 p-3 bg-slate-950/60 rounded-xl border border-slate-800/80 mb-4 text-xs">
        <div>
          <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold block">
            Actual
          </span>
          <span className="font-bold text-slate-200 text-sm">
            {formatValue(alert.actual_value)}
          </span>
        </div>
        <div>
          <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold block">
            Expected Baseline
          </span>
          <span className="font-bold text-slate-300 text-sm">
            {formatValue(alert.expected_value)}
          </span>
        </div>
        <div>
          <span className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold block">
            Deviation
          </span>
          <span
            className={`font-extrabold text-sm flex items-center space-x-1 ${
              isSpike ? 'text-emerald-400' : 'text-rose-400'
            }`}
          >
            {isSpike ? <TrendingUp className="w-3.5 h-3.5" /> : <TrendingDown className="w-3.5 h-3.5" />}
            <span>
              {isSpike ? '+' : ''}
              {alert.deviation_percent.toFixed(1)}%
            </span>
          </span>
        </div>
      </div>

      {/* Action Footer */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-800/70">
        {/* Workflow Triggers */}
        <div className="flex items-center space-x-2">
          <button
            type="button"
            onClick={() => onInvestigate?.(alert)}
            className="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-indigo-300 hover:bg-slate-700 hover:text-white transition-colors flex items-center space-x-1.5 border border-slate-700"
          >
            <Search className="w-3.5 h-3.5" />
            <span>Investigate</span>
          </button>
          <button
            type="button"
            onClick={() => onSimulate?.(alert)}
            className="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-emerald-300 hover:bg-slate-700 hover:text-white transition-colors flex items-center space-x-1.5 border border-slate-700"
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>Simulate</span>
          </button>
          <button
            type="button"
            onClick={() => onReview?.(alert)}
            className="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-amber-300 hover:bg-slate-700 hover:text-white transition-colors flex items-center space-x-1.5 border border-slate-700"
          >
            <ShieldCheck className="w-3.5 h-3.5" />
            <span>Review</span>
          </button>
        </div>

        {/* State Transitions */}
        <div className="flex items-center space-x-1.5">
          {alert.status === 'new' && (
            <button
              type="button"
              disabled={actionLoading}
              onClick={() => onAcknowledge?.(alert.id)}
              className="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-amber-950/60 text-amber-300 hover:bg-amber-900/80 border border-amber-800/70 transition-colors disabled:opacity-50"
            >
              Acknowledge
            </button>
          )}
          {(alert.status === 'new' || alert.status === 'acknowledged') && (
            <>
              <button
                type="button"
                disabled={actionLoading}
                onClick={() => onResolve?.(alert.id)}
                className="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-emerald-950/60 text-emerald-300 hover:bg-emerald-900/80 border border-emerald-800/70 transition-colors disabled:opacity-50"
              >
                Resolve
              </button>
              <button
                type="button"
                disabled={actionLoading}
                onClick={() => onDismiss?.(alert.id)}
                className="px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-slate-800/80 text-slate-400 hover:bg-slate-800 hover:text-slate-200 border border-slate-700 transition-colors disabled:opacity-50"
              >
                Dismiss
              </button>
            </>
          )}
          <button
            type="button"
            onClick={() => onSelect?.(alert)}
            className="px-2.5 py-1.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-indigo-400 transition-colors flex items-center space-x-1"
          >
            <Eye className="w-3.5 h-3.5" />
            <span>Detail</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default AlertCard;
