import React from 'react';
import { Database } from 'lucide-react';

const EmptyState = ({
  icon: Icon = Database,
  title = 'No records found',
  description = 'There are no items to display right now.',
  action,
}) => {
  return (
    <div className="flex flex-col items-center justify-center p-12 text-center border border-dashed border-slate-800 rounded-2xl bg-slate-900/30">
      <div className="p-3 bg-slate-800/50 rounded-xl text-slate-400 mb-3 border border-slate-700/50">
        <Icon className="w-8 h-8" />
      </div>
      <h3 className="text-base font-semibold text-slate-200">{title}</h3>
      <p className="text-xs text-slate-400 max-w-sm mt-1 mb-4">{description}</p>
      {action && <div>{action}</div>}
    </div>
  );
};

export default EmptyState;
