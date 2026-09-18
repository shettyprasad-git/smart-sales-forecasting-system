import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, ShoppingCart, Package, TrendingUp, ShieldAlert, Sliders, ShieldCheck, LogOut, X, Sparkles } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const navItems = [
  { path: '/dashboard', name: 'Dashboard', icon: LayoutDashboard },
  { path: '/sales', name: 'Sales Records', icon: ShoppingCart },
  { path: '/products', name: 'Products Catalog', icon: Package },
  { path: '/forecast', name: 'Demand Forecast', icon: TrendingUp },
  { path: '/anomalies', name: 'Anomaly Insights', icon: ShieldAlert },
  { path: '/simulation', name: 'What-If Simulation', icon: Sliders },
  { path: '/decisions', name: 'Decision Center', icon: ShieldCheck },
];

const Sidebar = ({ mobileOpen, setMobileOpen }) => {
  const { logout } = useAuth();

  const sidebarContent = (
    <div className="flex flex-col h-full bg-slate-900 border-r border-slate-800 text-slate-300 w-64 select-none">
      {/* Brand Header */}
      <div className="flex items-center justify-between h-16 px-6 border-b border-slate-800">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-gradient-to-tr from-indigo-600 to-indigo-400 rounded-xl shadow-lg shadow-indigo-950/50">
            <TrendingUp className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-sm font-bold text-slate-100 tracking-tight leading-tight m-0">
              Smart Sales
            </h1>
            <p className="text-[10px] text-indigo-400 font-semibold tracking-wider uppercase">
              Forecasting AI
            </p>
          </div>
        </div>
        {mobileOpen && (
          <button
            onClick={() => setMobileOpen(false)}
            className="p-1 rounded-lg text-slate-400 hover:text-white lg:hidden cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Navigation Links */}
      <div className="flex-1 px-4 py-6 space-y-1 overflow-y-auto custom-scrollbar">
        <div className="px-3 mb-2 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
          Main Navigation
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              onClick={() => setMobileOpen(false)}
              className={({ isActive }) =>
                `flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-xs font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-950/50 font-semibold'
                    : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/60'
                }`
              }
            >
              <Icon className="w-4 h-4" />
              <span>{item.name}</span>
            </NavLink>
          );
        })}
      </div>

      {/* System Status & Logout */}
      <div className="p-4 border-t border-slate-800 space-y-3">
        <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/60 flex items-center space-x-2.5">
          <Sparkles className="w-4 h-4 text-emerald-400 flex-shrink-0 animate-pulse" />
          <div>
            <div className="text-[11px] font-semibold text-slate-200">ML Engine Ready</div>
            <div className="text-[10px] text-slate-400">Random Forest / GB / LR</div>
          </div>
        </div>

        <button
          onClick={logout}
          className="flex items-center space-x-3 w-full px-3.5 py-2.5 rounded-xl text-xs font-medium text-slate-400 hover:text-rose-300 hover:bg-rose-950/30 border border-transparent hover:border-rose-900/40 transition-all cursor-pointer"
        >
          <LogOut className="w-4 h-4 text-slate-400 group-hover:text-rose-400" />
          <span>Log Out</span>
        </button>
      </div>
    </div>
  );

  return (
    <>
      {/* Desktop Sidebar */}
      <aside className="hidden lg:block fixed left-0 top-0 bottom-0 z-30">{sidebarContent}</aside>

      {/* Mobile Drawer Overlay */}
      {mobileOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div
            className="fixed inset-0 bg-slate-950/80 backdrop-blur-xs"
            onClick={() => setMobileOpen(false)}
          />
          <div className="fixed inset-y-0 left-0 z-50 w-64 shadow-2xl">{sidebarContent}</div>
        </div>
      )}
    </>
  );
};

export default Sidebar;
