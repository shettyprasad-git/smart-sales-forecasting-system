import React from 'react';
import { Link } from 'react-router-dom';
import { FileQuestion, ArrowLeft } from 'lucide-react';

const NotFound = () => {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center p-4">
      <div className="text-center space-y-4 max-w-md">
        <div className="inline-flex p-4 rounded-3xl bg-slate-900 border border-slate-800 text-indigo-400 mb-2 shadow-2xl">
          <FileQuestion className="w-12 h-12" />
        </div>
        <h1 className="text-3xl font-extrabold text-slate-100 tracking-tight">Page Not Found</h1>
        <p className="text-xs text-slate-400">
          The page URL you requested does not exist or has been moved.
        </p>
        <div>
          <Link
            to="/dashboard"
            className="inline-flex items-center space-x-2 px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-xl shadow-lg shadow-indigo-950/50 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            <span>Return to Executive Dashboard</span>
          </Link>
        </div>
      </div>
    </div>
  );
};

export default NotFound;
