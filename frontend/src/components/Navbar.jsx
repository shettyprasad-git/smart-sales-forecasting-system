import React from 'react';
import { Menu, User, LogOut } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const Navbar = ({ onToggleMobileMenu, title = 'Dashboard' }) => {
  const { user, logout } = useAuth();

  const userEmail = user?.email || 'User';
  const initial = userEmail.charAt(0).toUpperCase();

  return (
    <header className="sticky top-0 z-20 h-16 bg-slate-900/80 backdrop-blur-md border-b border-slate-800 px-3 sm:px-6 lg:px-8 flex items-center justify-between shrink-0">
      {/* Left side: Hamburger button + Page title */}
      <div className="flex items-center space-x-2 sm:space-x-3 min-w-0 pr-2">
        <button
          type="button"
          onClick={onToggleMobileMenu}
          className="p-2 -ml-1 rounded-xl text-slate-400 hover:text-slate-100 hover:bg-slate-800 lg:hidden cursor-pointer shrink-0 focus:outline-none focus:ring-2 focus:ring-indigo-500 min-h-[44px] min-w-[44px] flex items-center justify-center"
          aria-label="Open navigation menu"
        >
          <Menu className="w-5 h-5" />
        </button>
        <div className="min-w-0">
          <h2 className="text-sm sm:text-base lg:text-lg font-bold text-slate-100 tracking-tight m-0 truncate">
            {title}
          </h2>
        </div>
      </div>

      {/* Right side: User Profile & Actions */}
      <div className="flex items-center space-x-2 sm:space-x-4 shrink-0">
        <div className="flex items-center space-x-2 sm:space-x-3 pl-2 sm:pl-3 pr-1 py-1 rounded-full bg-slate-800/50 border border-slate-700/50 max-w-[160px] sm:max-w-[240px]">
          <div className="w-6 h-6 sm:w-7 sm:h-7 rounded-full bg-gradient-to-tr from-indigo-500 to-indigo-700 flex items-center justify-center text-[10px] sm:text-xs font-bold text-white shadow-xs shrink-0">
            {initial}
          </div>
          <span className="hidden sm:inline-block text-xs font-medium text-slate-300 truncate max-w-[140px]">
            {userEmail}
          </span>
          <button
            type="button"
            onClick={logout}
            aria-label="Log out"
            className="p-1.5 rounded-full text-slate-400 hover:text-rose-400 hover:bg-slate-800 transition-colors cursor-pointer shrink-0 focus:outline-none focus:ring-1 focus:ring-rose-500"
            title="Log Out"
          >
            <LogOut className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </header>
  );
};

export default Navbar;
