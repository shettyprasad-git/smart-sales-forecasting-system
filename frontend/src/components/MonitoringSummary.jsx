import React from 'react';
import { Bell, AlertOctagon, AlertTriangle, ShieldAlert, CheckCircle2, Clock, CopyCheck } from 'lucide-react';

const MonitoringSummary = ({ summary, loading }) => {
  if (!summary && !loading) return null;

  const kpis = [
    {
      title: 'New Alerts',
      value: summary?.new_alerts ?? 0,
      icon: Bell,
      color: 'text-indigo-400',
      bg: 'bg-indigo-950/40 border-indigo-800/60',
      badge: 'Unacknowledged',
      badgeColor: 'bg-indigo-900/50 text-indigo-300 border-indigo-700/50',
    },
    {
      title: 'Critical Attention',
      value: summary?.critical_alerts ?? 0,
      icon: AlertOctagon,
      color: 'text-rose-400',
      bg: 'bg-rose-950/40 border-rose-800/60',
      badge: 'Immediate Action',
      badgeColor: 'bg-rose-900/50 text-rose-300 border-rose-700/50',
    },
    {
      title: 'High Priority',
      value: summary?.high_alerts ?? 0,
      icon: AlertTriangle,
      color: 'text-amber-400',
      bg: 'bg-amber-950/40 border-amber-800/60',
      badge: 'Review Needed',
      badgeColor: 'bg-amber-900/50 text-amber-300 border-amber-700/50',
    },
    {
      title: 'Active Unresolved',
      value: summary?.unresolved_alerts ?? 0,
      icon: ShieldAlert,
      color: 'text-sky-400',
      bg: 'bg-sky-950/40 border-sky-800/60',
      badge: 'Open Queue',
      badgeColor: 'bg-sky-900/50 text-sky-300 border-sky-700/50',
    },
  ];

  return (
    <div className="space-y-4">
      {/* Top 4 KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {kpis.map((kpi, idx) => {
          const Icon = kpi.icon;
          return (
            <div
              key={idx}
              className={`p-5 rounded-2xl border transition-all duration-200 shadow-lg ${kpi.bg}`}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center space-x-2.5">
                  <div className={`p-2 rounded-xl bg-slate-900/80 ${kpi.color}`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <span className="text-xs font-semibold text-slate-300 tracking-wide">
                    {kpi.title}
                  </span>
                </div>
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${kpi.badgeColor}`}>
                  {kpi.badge}
                </span>
              </div>
              <div className="flex items-baseline justify-between mt-2">
                <span className="text-3xl font-extrabold text-white tracking-tight">
                  {loading ? '—' : kpi.value}
                </span>
                <span className="text-xs text-slate-400">Total alerts</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Observability Telemetry Strip */}
      {summary && (
        <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-2.5 bg-slate-900/60 border border-slate-800 rounded-xl text-xs text-slate-400">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-1.5">
              <Clock className="w-3.5 h-3.5 text-indigo-400" />
              <span>
                Last Scan:{' '}
                <strong className="text-slate-200">
                  {new Date(summary.scan_timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                </strong>
              </span>
            </div>
            <div className="hidden sm:inline-block text-slate-700">•</div>
            <div className="flex items-center space-x-1.5">
              <CopyCheck className="w-3.5 h-3.5 text-emerald-400" />
              <span>
                Duplicates Suppressed:{' '}
                <strong className="text-slate-200">{summary.suppressed_duplicates}</strong>
              </span>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-950/60 text-emerald-400 border border-emerald-800/60">
              <CheckCircle2 className="w-3 h-3 mr-1" /> Monitoring Active
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default MonitoringSummary;
