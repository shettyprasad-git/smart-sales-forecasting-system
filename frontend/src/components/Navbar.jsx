import React from 'react';
import { Menu, User, LogOut } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

const Navbar = ({ onToggleMobileMenu, title = 'Dashboard' }) => {
  const { user, logout } = useAuth();

  const userEmail = user?.email || 'User';
  const initial = userEmail.charAt(0).toUpperCase();

  return (
    <header className="sticky top-0 z-20 h-16 bg-slate-900/80 backdrop-blur-md border-b border-slate-800 px-4 lg:px-8 flex items-center justify-between">
      {/* Left side: Hamburger button + Page title */}
      <div className="flex items-center space-x-3">
        <button
          onClick={onToggleMobileMenu}
          className="p-2 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 lg:hidden cursor-pointer"
          aria-label="Toggle menu"
        >
          <Menu className="w-5 h-5" />
        </button>
        <div className="flex flex-col">
          <h2 className="text-base lg:text-lg font-bold text-slate-100 tracking-tight m-0">
            {title}
          </h2>
        </div>
      </div>

      {/* Right side: User Profile & Actions */}
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-3 pl-3 pr-1 py-1 rounded-full bg-slate-800/50 border border-slate-700/50">
          <div className="w-7 h-7 rounded-full bg-gradient-to-tr from-indigo-500 to-indigo-700 flex items-center justify-center text-xs font-bold text-white shadow-xs">
            {initial}
          </div>
          <span className="hidden sm:inline-block text-xs font-medium text-slate-300 max-w-[180px] truncate">
            {userEmail}
          </span>
          <button
            onClick={logout}
            className="p-1.5 rounded-full text-slate-400 hover:text-rose-400 hover:bg-slate-800 transition-colors cursor-pointer"
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
