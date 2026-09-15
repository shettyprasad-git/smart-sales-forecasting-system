import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

const ErrorMessage = ({
  title = 'An error occurred',
  message = 'Failed to load data. Please check your network connection or try again.',
  onRetry,
}) => {
  return (
    <div className="rounded-xl border border-rose-500/20 bg-rose-950/20 p-5 text-rose-200 backdrop-blur-sm shadow-lg my-4">
      <div className="flex items-start space-x-3">
        <AlertCircle className="w-5 h-5 text-rose-400 mt-0.5 flex-shrink-0" />
        <div className="flex-1">
          <h4 className="text-sm font-semibold text-rose-300">{title}</h4>
          <p className="text-xs text-rose-200/80 mt-1 leading-relaxed">{message}</p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-3 inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 border border-rose-500/30 text-xs font-medium transition-colors cursor-pointer"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retry Request</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default ErrorMessage;
