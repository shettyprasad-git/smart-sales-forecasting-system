import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import {
  TrendingUp,
  ShieldAlert,
  Sliders,
  ShieldCheck,
  Eye,
  EyeOff,
  Lock,
  Mail,
  Loader2,
  ArrowRight,
  Sparkles,
  CheckCircle2,
} from 'lucide-react';
import { extractErrorMessage } from '../api/axios';

const CAPABILITIES = [
  {
    icon: TrendingUp,
    title: 'Forecast Demand',
    description:
      'Generate 7-day, 30-day, and 90-day demand forecasts using company-specific models.',
    color: 'text-indigo-400',
    bg: 'bg-indigo-500/10 border-indigo-500/20',
  },
  {
    icon: ShieldAlert,
    title: 'Detect & Investigate Anomalies',
    description:
      'Identify unusual sales behavior and investigate associated contributors.',
    color: 'text-amber-400',
    bg: 'bg-amber-500/10 border-amber-500/20',
  },
  {
    icon: Sliders,
    title: 'Run What-If Scenarios',
    description:
      'Explore demand, price, discount, promotion, and other supported scenarios using dedicated models.',
    color: 'text-emerald-400',
    bg: 'bg-emerald-500/10 border-emerald-500/20',
  },
  {
    icon: ShieldCheck,
    title: 'Govern Decisions',
    description:
      'Keep recommendations and business actions behind explicit human approval.',
    color: 'text-purple-400',
    bg: 'bg-purple-500/10 border-purple-500/20',
  },
];

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const { login } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    const notice = sessionStorage.getItem('auth_notice');
    const searchParams = new URLSearchParams(window.location.search);
    if (notice || searchParams.get('expired')) {
      setError(notice || 'Your session has expired. Please sign in again.');
      sessionStorage.removeItem('auth_notice');
    }
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (!email || !password) {
      setError('Please enter both email and password.');
      return;
    }

    try {
      setIsSubmitting(true);
      await login(email, password);
      navigate('/dashboard');
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-[100dvh] bg-slate-950 text-slate-100 flex items-center justify-center p-3 sm:p-6 lg:p-10 relative selection:bg-indigo-500 selection:text-white overflow-x-hidden">
      {/* Subtle Background Glows */}
      <div className="absolute top-1/4 left-1/4 -translate-x-1/2 -translate-y-1/2 w-72 sm:w-96 h-72 sm:h-96 bg-indigo-600/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 translate-x-1/2 translate-y-1/2 w-72 sm:w-96 h-72 sm:h-96 bg-purple-600/10 rounded-full blur-3xl pointer-events-none" />

      <div className="w-full max-w-6xl relative z-10 my-auto">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-12 items-center">
          {/* ============================================================== */}
          {/* LEFT PANEL: Product Introduction & Capability Overview         */}
          {/* On Desktop/Tablet: Rich 2-column presentation                  */}
          {/* On Mobile: Compact, structured banner above login form         */}
          {/* ============================================================== */}
          <div className="lg:col-span-7 space-y-4 sm:space-y-6">
            {/* Brand Header & Headline */}
            <div className="space-y-2.5 sm:space-y-3">
              <div className="flex items-center space-x-2.5">
                <div className="p-2 sm:p-2.5 bg-gradient-to-tr from-indigo-600 to-indigo-400 rounded-xl shadow-lg shadow-indigo-950/60 shrink-0">
                  <TrendingUp className="w-5 h-5 sm:w-6 sm:h-6 text-white" />
                </div>
                <div>
                  <span className="text-[10px] sm:text-xs font-bold uppercase tracking-wider text-indigo-400 block font-mono">
                    Decision Intelligence Platform
                  </span>
                  <h1 className="text-xl sm:text-2xl lg:text-3xl font-extrabold tracking-tight text-slate-100 leading-tight">
                    Smart Sales Forecasting System
                  </h1>
                </div>
              </div>

              <p className="text-xs sm:text-sm text-slate-300/90 leading-relaxed max-w-xl">
                AI-assisted sales forecasting and decision intelligence for demand planning, anomaly investigation, and what-if analysis.
              </p>
            </div>

            {/* Desktop & Tablet: Full 4 Capability Cards */}
            <div className="hidden sm:grid sm:grid-cols-2 gap-3.5 pt-1">
              {CAPABILITIES.map((cap, idx) => {
                const Icon = cap.icon;
                return (
                  <div
                    key={idx}
                    className="p-3.5 sm:p-4 rounded-2xl bg-slate-900/80 border border-slate-800/90 shadow-md hover:border-slate-700/80 transition-all space-y-1.5"
                  >
                    <div className="flex items-center space-x-2">
                      <div className={`p-1.5 rounded-lg border ${cap.bg} ${cap.color}`}>
                        <Icon className="w-4 h-4" />
                      </div>
                      <h3 className="text-xs sm:text-sm font-bold text-slate-100">
                        {cap.title}
                      </h3>
                    </div>
                    <p className="text-[11px] sm:text-xs text-slate-400 leading-relaxed">
                      {cap.description}
                    </p>
                  </div>
                );
              })}
            </div>

            {/* Mobile Only: Compact Capability Pills (Takes minimal vertical space) */}
            <div className="sm:hidden grid grid-cols-2 gap-2 pt-0.5">
              {CAPABILITIES.map((cap, idx) => {
                const Icon = cap.icon;
                return (
                  <div
                    key={idx}
                    className="p-2 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center space-x-2"
                  >
                    <div className={`p-1 rounded-md border ${cap.bg} ${cap.color} shrink-0`}>
                      <Icon className="w-3.5 h-3.5" />
                    </div>
                    <span className="text-[11px] font-semibold text-slate-200 truncate">
                      {cap.title}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* Trust & Boundary Notice */}
            <div className="hidden sm:flex items-center space-x-2 text-[11px] text-slate-400 font-medium pt-1">
              <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>Human-in-the-loop governance: Zero autonomous business actions are executed without explicit approval.</span>
            </div>
          </div>

          {/* ============================================================== */}
          {/* RIGHT PANEL: Sign In Card                                      */}
          {/* ============================================================== */}
          <div className="lg:col-span-5 w-full">
            <div className="bg-slate-900/95 border border-slate-800 rounded-2xl sm:rounded-3xl p-5 sm:p-7 md:p-8 shadow-2xl backdrop-blur-md space-y-5">
              <div className="border-b border-slate-800 pb-3.5">
                <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-400 block font-mono mb-1">
                  Executive Workspace
                </span>
                <h2 className="text-lg sm:text-xl font-bold text-slate-100">
                  Sign In
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Access your forecasting and decision intelligence workspace.
                </p>
              </div>

              {error && (
                <div
                  role="alert"
                  className="p-3.5 rounded-xl bg-rose-950/40 border border-rose-500/30 text-rose-300 text-xs flex items-start space-x-2.5 leading-relaxed break-words"
                >
                  <span className="font-medium">{error}</span>
                </div>
              )}

              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label
                    htmlFor="login-email"
                    className="block text-xs font-semibold text-slate-300 mb-1.5"
                  >
                    Email Address
                  </label>
                  <div className="relative">
                    <Mail className="w-4 h-4 text-slate-500 absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" />
                    <input
                      id="login-email"
                      name="email"
                      type="email"
                      autoComplete="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="name@company.com"
                      required
                      className="w-full pl-10 pr-4 py-2.5 sm:py-3 bg-slate-950 border border-slate-800 rounded-xl text-xs sm:text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all"
                    />
                  </div>
                </div>

                <div>
                  <label
                    htmlFor="login-password"
                    className="block text-xs font-semibold text-slate-300 mb-1.5"
                  >
                    Password
                  </label>
                  <div className="relative">
                    <Lock className="w-4 h-4 text-slate-500 absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" />
                    <input
                      id="login-password"
                      name="password"
                      type={showPassword ? 'text' : 'password'}
                      autoComplete="current-password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="••••••••"
                      required
                      className="w-full pl-10 pr-11 py-2.5 sm:py-3 bg-slate-950 border border-slate-800 rounded-xl text-xs sm:text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all"
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword(!showPassword)}
                      aria-label={showPassword ? 'Hide password' : 'Show password'}
                      className="absolute right-3.5 top-1/2 -translate-y-1/2 p-1 text-slate-500 hover:text-slate-300 transition-colors cursor-pointer rounded-lg focus:outline-none focus:ring-1 focus:ring-indigo-500"
                    >
                      {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>

                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="w-full py-3 px-4 bg-indigo-600 hover:bg-indigo-500 text-white text-xs sm:text-sm font-bold rounded-xl shadow-lg shadow-indigo-950/50 transition-all duration-150 flex items-center justify-center space-x-2 cursor-pointer disabled:opacity-50 min-h-[44px]"
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin shrink-0" />
                      <span>Signing In...</span>
                    </>
                  ) : (
                    <>
                      <span>Sign In</span>
                      <ArrowRight className="w-4 h-4 shrink-0" />
                    </>
                  )}
                </button>
              </form>

              <div className="pt-2 text-center border-t border-slate-800/80">
                <p className="text-xs text-slate-400">
                  Don't have an account?{' '}
                  <Link
                    to="/register"
                    className="font-semibold text-indigo-400 hover:text-indigo-300 underline underline-offset-4 transition-colors"
                  >
                    Create Account
                  </Link>
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
