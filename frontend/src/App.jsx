import React, { lazy, Suspense } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import DashboardLayout from './layouts/DashboardLayout';
import LoadingSpinner from './components/LoadingSpinner';

const Login = lazy(() => import('./pages/Login'));
const Register = lazy(() => import('./pages/Register'));
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Products = lazy(() => import('./pages/Products'));
const Sales = lazy(() => import('./pages/Sales'));
const Forecast = lazy(() => import('./pages/Forecast'));
const Anomalies = lazy(() => import('./pages/Anomalies'));
const Simulation = lazy(() => import('./pages/Simulation'));
const DecisionCenter = lazy(() => import('./pages/DecisionCenter'));
const IntelligenceMonitor = lazy(() => import('./pages/IntelligenceMonitor'));
const NotFound = lazy(() => import('./pages/NotFound'));

// Root redirect handler
const RootRedirect = () => {
  const { isAuthenticated, loading } = useAuth();
  if (loading) return null;
  return isAuthenticated ? <Navigate to="/dashboard" replace /> : <Navigate to="/login" replace />;
};

const AppContent = () => {
  return (
    <Suspense
      fallback={
        <div className="flex h-screen items-center justify-center bg-slate-950">
          <LoadingSpinner size="lg" text="Loading executive workspace..." />
        </div>
      }
    >
      <Routes>
        {/* Public Auth Routes */}
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />

        {/* Root redirect */}
        <Route path="/" element={<RootRedirect />} />

        {/* Guarded Executive Workspace Routes */}
        <Route element={<ProtectedRoute />}>
          <Route element={<DashboardLayout />}>
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/products" element={<Products />} />
            <Route path="/sales" element={<Sales />} />
            <Route path="/forecast" element={<Forecast />} />
            <Route path="/anomalies" element={<Anomalies />} />
            <Route path="/simulation" element={<Simulation />} />
            <Route path="/decisions" element={<DecisionCenter />} />
            <Route path="/decisions/:id" element={<DecisionCenter />} />
            <Route path="/intelligence" element={<IntelligenceMonitor />} />
          </Route>
        </Route>

        {/* Fallback 404 Route */}
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Suspense>
  );
};

function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;
