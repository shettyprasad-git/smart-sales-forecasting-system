import React, { useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import Sidebar from '../components/Sidebar';
import Navbar from '../components/Navbar';

const pageTitles = {
  '/dashboard': 'Executive Sales Dashboard',
  '/datasets': 'Dataset Management & Runtime Pipeline',
  '/sales': 'Sales Management',
  '/products': 'Products Catalog',
  '/forecast': 'Demand Forecast Engine',
  '/anomalies': 'Sales Anomaly Detection',
  '/simulation': 'What-If Scenario Simulation',
  '/decisions': 'Executive Decision Center',
  '/intelligence': 'Intelligence Monitoring & Health',
};

const DashboardLayout = () => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  const title = pageTitles[location.pathname] || 'Smart Sales Forecasting';

  return (
    <div className="min-h-[100dvh] bg-slate-950 text-slate-100 flex flex-col font-sans antialiased overflow-x-hidden">
      {/* Sidebar navigation */}
      <Sidebar mobileOpen={mobileOpen} setMobileOpen={setMobileOpen} />

      {/* Main content wrapper */}
      <div className="lg:pl-64 flex flex-col flex-1 min-h-[100dvh] min-w-0">
        <Navbar onToggleMobileMenu={() => setMobileOpen(!mobileOpen)} title={title} />
        <main className="flex-1 p-3.5 sm:p-5 lg:p-8 max-w-7xl w-full mx-auto min-w-0 overflow-x-hidden">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default DashboardLayout;
