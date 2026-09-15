import React from 'react';

const colorThemes = {
  indigo: {
    iconBg: 'bg-indigo-500/10 border-indigo-500/20 text-indigo-400',
    glow: 'hover:border-indigo-500/40 hover:shadow-indigo-950/30',
  },
  emerald: {
    iconBg: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400',
    glow: 'hover:border-emerald-500/40 hover:shadow-emerald-950/30',
  },
  amber: {
    iconBg: 'bg-amber-500/10 border-amber-500/20 text-amber-400',
    glow: 'hover:border-amber-500/40 hover:shadow-amber-950/30',
  },
  sky: {
    iconBg: 'bg-sky-500/10 border-sky-500/20 text-sky-400',
    glow: 'hover:border-sky-500/40 hover:shadow-sky-950/30',
  },
};

const KpiCard = ({ title, value, icon: Icon, description, color = 'indigo' }) => {
  const theme = colorThemes[color] || colorThemes.indigo;

  return (
    <div
      className={`relative rounded-2xl bg-slate-900 border border-slate-800/80 p-5 shadow-lg transition-all duration-200 ${theme.glow} group overflow-hidden`}
    >
      {/* Background Accent Pill */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          {title}
        </span>
        <div className={`p-2.5 rounded-xl border ${theme.iconBg} transition-transform group-hover:scale-105`}>
          <Icon className="w-4 h-4" />
        </div>
      </div>

      <div className="space-y-1">
        <div className="text-2xl font-bold tracking-tight text-slate-100 font-mono">
          {value}
        </div>
        {description && (
          <p className="text-[11px] font-medium text-slate-400 leading-snug">
            {description}
          </p>
        )}
      </div>
    </div>
  );
};

export default KpiCard;
