import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import {
  TrendingUp,
  ShieldAlert,
  Sliders,
  ShieldCheck,
  Mail,
  Lock,
  Loader2,
  UserPlus,
  CheckCircle2,
  Eye,
  EyeOff,
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

const Register = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [successMsg, setSuccessMsg] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const { register } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccessMsg('');

    if (!email || !password || !confirmPassword) {
      setError('Please fill in all required fields.');
      return;
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match. Please verify your password entry.');
      return;
    }

    if (password.length < 6) {
      setError('Password must be at least 6 characters long.');
      return;
    }

    try {
      setIsSubmitting(true);
      await register(email, password);
      setSuccessMsg('Account created successfully! Redirecting to sign in...');
      setTimeout(() => {
        navigate('/login');
      }, 1500);
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
          {/* LEFT PANEL: Platform Intro */}
          <div className="lg:col-span-7 space-y-4 sm:space-y-6">
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

            {/* Desktop & Tablet Capabilities */}
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

            {/* Mobile Only: Compact Pills */}
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

            <div className="hidden sm:flex items-center space-x-2 text-[11px] text-slate-400 font-medium pt-1">
              <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>Multi-tenant isolation and complete enterprise privacy.</span>
            </div>
          </div>

          {/* RIGHT PANEL: Register Card */}
          <div className="lg:col-span-5 w-full">
            <div className="bg-slate-900/95 border border-slate-800 rounded-2xl sm:rounded-3xl p-5 sm:p-7 md:p-8 shadow-2xl backdrop-blur-md space-y-5">
              <div className="border-b border-slate-800 pb-3.5">
                <span className="text-[10px] font-bold uppercase tracking-wider text-indigo-400 block font-mono mb-1">
                  Tenant Registration
                </span>
                <h2 className="text-lg sm:text-xl font-bold text-slate-100">
                  Create Account
                </h2>
                <p className="text-xs text-slate-400 mt-0.5">
                  Get started with sales forecasting and decision intelligence.
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

              {successMsg && (
                <div
                  role="status"
                  className="p-3.5 rounded-xl bg-emerald-950/40 border border-emerald-500/30 text-emerald-300 text-xs flex items-center space-x-2 leading-relaxed"
                >
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                  <span className="font-medium">{successMsg}</span>
                </div>
              )}

              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <label
                    htmlFor="reg-email"
                    className="block text-xs font-semibold text-slate-300 mb-1.5"
                  >
                    Email Address
                  </label>
                  <div className="relative">
                    <Mail className="w-4 h-4 text-slate-500 absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" />
                    <input
                      id="reg-email"
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
                    htmlFor="reg-password"
                    className="block text-xs font-semibold text-slate-300 mb-1.5"
                  >
                    Password
                  </label>
                  <div className="relative">
                    <Lock className="w-4 h-4 text-slate-500 absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" />
                    <input
                      id="reg-password"
                      name="password"
                      type={showPassword ? 'text' : 'password'}
                      autoComplete="new-password"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="At least 6 characters"
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

                <div>
                  <label
                    htmlFor="reg-confirm-password"
                    className="block text-xs font-semibold text-slate-300 mb-1.5"
                  >
                    Confirm Password
                  </label>
                  <div className="relative">
                    <Lock className="w-4 h-4 text-slate-500 absolute left-3.5 top-1/2 -translate-y-1/2 pointer-events-none" />
                    <input
                      id="reg-confirm-password"
                      name="confirmPassword"
                      type={showPassword ? 'text' : 'password'}
                      autoComplete="new-password"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      placeholder="Re-enter password"
                      required
                      className="w-full pl-10 pr-4 py-2.5 sm:py-3 bg-slate-950 border border-slate-800 rounded-xl text-xs sm:text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all"
                    />
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
                      <span>Creating Account...</span>
                    </>
                  ) : (
                    <>
                      <UserPlus className="w-4 h-4 shrink-0" />
                      <span>Register Account</span>
                    </>
                  )}
                </button>
              </form>

              <div className="pt-2 text-center border-t border-slate-800/80">
                <p className="text-xs text-slate-400">
                  Already registered?{' '}
                  <Link
                    to="/login"
                    className="font-semibold text-indigo-400 hover:text-indigo-300 underline underline-offset-4 transition-colors"
                  >
                    Sign In
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

export default Register;
