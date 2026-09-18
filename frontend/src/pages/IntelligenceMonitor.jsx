import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  runMonitoring,
  getAlerts,
  getAlert,
  acknowledgeAlert,
  resolveAlert,
  dismissAlert,
  getMonitoringSummary,
} from '../api/monitoring';
import MonitoringSummary from '../components/MonitoringSummary';
import AlertCard from '../components/AlertCard';
import InvestigationDrawer from '../components/InvestigationDrawer';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import EmptyState from '../components/EmptyState';
import {
  Activity,
  Play,
  RotateCw,
  Filter,
  CheckCircle2,
  AlertOctagon,
  ShieldCheck,
  Sparkles,
  Sliders,
  Search,
  X,
  ChevronRight,
  Info,
} from 'lucide-react';

const STATUS_TABS = [
  { id: 'all', label: 'All Alerts' },
  { id: 'new', label: 'New' },
  { id: 'acknowledged', label: 'Acknowledged' },
  { id: 'resolved', label: 'Resolved' },
  { id: 'dismissed', label: 'Dismissed' },
];

const SEVERITIES = ['all', 'critical', 'high', 'medium', 'low'];

const ALERT_TYPES = [
  { id: 'all', label: 'All Types' },
  { id: 'sales_spike', label: 'Sales Spike' },
  { id: 'sales_drop', label: 'Sales Drop' },
  { id: 'repeated_anomaly', label: 'Repeated Anomaly' },
  { id: 'category_deviation', label: 'Category Deviation' },
  { id: 'product_deviation', label: 'Product Deviation' },
  { id: 'demand_shift', label: 'Demand Shift' },
  { id: 'trend_change', label: 'Trend Change' },
];

const IntelligenceMonitor = () => {
  const navigate = useNavigate();

  const [alerts, setAlerts] = useState([]);
  const [totalAlerts, setTotalAlerts] = useState(0);
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [scanning, setScanning] = useState(false);
  const [scanResult, setScanResult] = useState(null);
  const [error, setError] = useState(null);

  // Filters
  const [statusFilter, setStatusFilter] = useState('all');
  const [severityFilter, setSeverityFilter] = useState('all');
  const [alertTypeFilter, setAlertTypeFilter] = useState('all');
  const [page, setPage] = useState(0);
  const limit = 20;

  // Selected Alert for Deep Drawer
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [investigationAnomalyId, setInvestigationAnomalyId] = useState(null);
  const [actionLoading, setActionLoading] = useState(false);

  // Fetch summary and alerts
  const loadData = useCallback(async () => {
    try {
      setError(null);
      const [summaryData, alertsData] = await Promise.all([
        getMonitoringSummary(),
        getAlerts({
          status: statusFilter === 'all' ? undefined : statusFilter,
          severity: severityFilter === 'all' ? undefined : severityFilter,
          alert_type: alertTypeFilter === 'all' ? undefined : alertTypeFilter,
          skip: page * limit,
          limit,
        }),
      ]);
      setSummary(summaryData);
      setAlerts(alertsData.items);
      setTotalAlerts(alertsData.total);
    } catch (err) {
      console.error('Failed to load monitoring data:', err);
      setError(err.response?.data?.detail || 'Failed to load intelligence monitoring feed.');
    } finally {
      setLoading(false);
    }
  }, [statusFilter, severityFilter, alertTypeFilter, page, limit]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Trigger manual monitoring scan
  const handleRunScan = async () => {
    setScanning(true);
    setScanResult(null);
    setError(null);
    try {
      const res = await runMonitoring({ lookback_days: 30, min_severity: 'medium' });
      setScanResult({
        alertsCreated: res.alerts_created,
        duplicatesSuppressed: res.duplicates_suppressed,
        timestamp: new Date().toLocaleTimeString(),
      });
      await loadData();
    } catch (err) {
      console.error('Monitoring scan failed:', err);
      setError(err.response?.data?.detail || 'Monitoring scan failed to complete.');
    } finally {
      setScanning(false);
    }
  };

  // State transitions
  const handleAcknowledge = async (alertId) => {
    setActionLoading(true);
    try {
      await acknowledgeAlert(alertId);
      await loadData();
      if (selectedAlert && selectedAlert.id === alertId) {
        const updated = await getAlert(alertId);
        setSelectedAlert(updated);
      }
    } catch (err) {
      console.error('Failed to acknowledge alert:', err);
      alert(err.response?.data?.detail || 'Failed to acknowledge alert');
    } finally {
      setActionLoading(false);
    }
  };

  const handleResolve = async (alertId) => {
    setActionLoading(true);
    try {
      await resolveAlert(alertId);
      await loadData();
      if (selectedAlert && selectedAlert.id === alertId) {
        const updated = await getAlert(alertId);
        setSelectedAlert(updated);
      }
    } catch (err) {
      console.error('Failed to resolve alert:', err);
      alert(err.response?.data?.detail || 'Failed to resolve alert');
    } finally {
      setActionLoading(false);
    }
  };

  const handleDismiss = async (alertId) => {
    setActionLoading(true);
    try {
      await dismissAlert(alertId);
      await loadData();
      if (selectedAlert && selectedAlert.id === alertId) {
        const updated = await getAlert(alertId);
        setSelectedAlert(updated);
      }
    } catch (err) {
      console.error('Failed to dismiss alert:', err);
      alert(err.response?.data?.detail || 'Failed to dismiss alert');
    } finally {
      setActionLoading(false);
    }
  };

  // Open detail drawer
  const handleOpenDetail = async (alertItem) => {
    setDetailLoading(true);
    try {
      const detail = await getAlert(alertItem.id);
      setSelectedAlert(detail);
    } catch (err) {
      console.error('Failed to load alert detail:', err);
      setSelectedAlert(alertItem);
    } finally {
      setDetailLoading(false);
    }
  };

  // Navigate to What-If Simulation
  const handleSimulate = (alertItem) => {
    navigate('/simulation', {
      state: {
        anomalyId: alertItem.anomaly_id,
        horizon: 30,
      },
    });
  };

  // Navigate to Decision Center
  const handleReview = () => {
    navigate('/decisions');
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2.5">
            <div className="p-2 bg-gradient-to-tr from-indigo-600 to-indigo-400 rounded-xl shadow-lg shadow-indigo-950/50">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-2xl font-black text-slate-100 tracking-tight">
              Intelligence Monitor
            </h1>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            Proactive commercial surveillance, demand shift detection, and duplicate-suppressed alerts.
          </p>
        </div>

        {/* Actions */}
        <div className="flex items-center space-x-3">
          <button
            type="button"
            disabled={scanning}
            onClick={handleRunScan}
            className="px-4 py-2.5 rounded-xl font-bold text-xs bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-indigo-400 text-white shadow-lg shadow-indigo-950/50 transition-all flex items-center space-x-2 disabled:opacity-50 cursor-pointer"
          >
            {scanning ? (
              <>
                <RotateCw className="w-4 h-4 animate-spin" />
                <span>Scanning Sales Data...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>Run Monitoring Scan</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Scan Feedback Banner */}
      {scanResult && (
        <div className="p-4 rounded-xl bg-indigo-950/50 border border-indigo-800/80 text-indigo-300 text-xs flex items-center justify-between shadow-lg">
          <div className="flex items-center space-x-2.5">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
            <span>
              Scan completed at {scanResult.timestamp}:{' '}
              <strong className="text-white">{scanResult.alertsCreated}</strong> new alerts created,{' '}
              <strong className="text-white">{scanResult.duplicatesSuppressed}</strong> duplicates suppressed.
            </span>
          </div>
          <button
            onClick={() => setScanResult(null)}
            className="text-slate-400 hover:text-white"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Error Message */}
      {error && <ErrorMessage message={error} onDismiss={() => setError(null)} />}

      {/* Top KPI Telemetry */}
      <MonitoringSummary summary={summary} loading={loading} />

      {/* Filter Toolbar */}
      <div className="p-4 bg-slate-900/60 border border-slate-800 rounded-2xl space-y-4">
        {/* Status Tabs */}
        <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 pb-3">
          {STATUS_TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => {
                setStatusFilter(tab.id);
                setPage(0);
              }}
              className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition-all cursor-pointer ${
                statusFilter === tab.id
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-950/50'
                  : 'bg-slate-800/60 text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Dropdown Filters */}
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center space-x-2">
            <span className="text-xs text-slate-400 font-medium">Severity:</span>
            <select
              value={severityFilter}
              onChange={(e) => {
                setSeverityFilter(e.target.value);
                setPage(0);
              }}
              className="px-3 py-1.5 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-200 focus:outline-none focus:border-indigo-500 capitalize"
            >
              {SEVERITIES.map((sev) => (
                <option key={sev} value={sev}>
                  {sev === 'all' ? 'All Severities' : sev}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center space-x-2">
            <span className="text-xs text-slate-400 font-medium">Alert Type:</span>
            <select
              value={alertTypeFilter}
              onChange={(e) => {
                setAlertTypeFilter(e.target.value);
                setPage(0);
              }}
              className="px-3 py-1.5 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              {ALERT_TYPES.map((type) => (
                <option key={type.id} value={type.id}>
                  {type.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Feed Content */}
      {loading ? (
        <div className="py-16 flex justify-center">
          <LoadingSpinner />
        </div>
      ) : alerts.length === 0 ? (
        <EmptyState
          title="No alerts match criteria"
          description="Click 'Run Monitoring Scan' above to scan recent sales data for anomalies and commercial shifts."
          actionText="Run Scan Now"
          onAction={handleRunScan}
        />
      ) : (
        <div className="space-y-4">
          <div className="flex items-center justify-between text-xs text-slate-400 px-1">
            <span>
              Showing <strong className="text-slate-200">{alerts.length}</strong> of{' '}
              <strong className="text-slate-200">{totalAlerts}</strong> alerts
            </span>
          </div>

          {/* Alert Cards Feed */}
          <div className="grid grid-cols-1 gap-4">
            {alerts.map((alert) => (
              <AlertCard
                key={alert.id}
                alert={alert}
                actionLoading={actionLoading}
                onSelect={handleOpenDetail}
                onAcknowledge={handleAcknowledge}
                onResolve={handleResolve}
                onDismiss={handleDismiss}
                onInvestigate={() => setInvestigationAnomalyId(alert.anomaly_id)}
                onSimulate={handleSimulate}
                onReview={handleReview}
              />
            ))}
          </div>
        </div>
      )}

      {/* Alert Detail Slide-Over / Modal */}
      {selectedAlert && (
        <div className="fixed inset-0 z-50 overflow-hidden bg-slate-950/80 backdrop-blur-sm flex justify-end">
          <div className="w-full max-w-xl bg-slate-900 border-l border-slate-800 h-full overflow-y-auto p-6 space-y-6 shadow-2xl flex flex-col justify-between">
            <div className="space-y-6">
              {/* Drawer Header */}
              <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                <div>
                  <span className="text-[10px] font-bold text-indigo-400 uppercase tracking-widest block">
                    Alert Intelligence Details
                  </span>
                  <h2 className="text-lg font-bold text-white mt-1">
                    {selectedAlert.title}
                  </h2>
                </div>
                <button
                  type="button"
                  onClick={() => setSelectedAlert(null)}
                  className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              {/* Governance Advisory Banner */}
              <div className="p-3.5 rounded-xl bg-amber-950/40 border border-amber-800/60 text-amber-300 text-xs flex items-start space-x-2.5">
                <ShieldCheck className="w-4 h-4 flex-shrink-0 mt-0.5 text-amber-400" />
                <div>
                  <strong className="font-semibold block">Human Review Required</strong>
                  <span>
                    Alerts surface statistical deviations for human consideration only. No automated price changes, purchase orders, or inventory allocations are executed.
                  </span>
                </div>
              </div>

              {/* Detail Metrics */}
              <div className="grid grid-cols-3 gap-3 p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs">
                <div>
                  <span className="text-slate-500 block text-[10px] uppercase font-semibold">Actual</span>
                  <span className="text-sm font-bold text-slate-200">
                    {selectedAlert.metric === 'sales_amount' ? `₹${selectedAlert.actual_value.toLocaleString()}` : `${selectedAlert.actual_value.toLocaleString()} units`}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px] uppercase font-semibold">Baseline</span>
                  <span className="text-sm font-bold text-slate-300">
                    {selectedAlert.metric === 'sales_amount' ? `₹${selectedAlert.expected_value.toLocaleString()}` : `${selectedAlert.expected_value.toLocaleString()} units`}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500 block text-[10px] uppercase font-semibold">Deviation</span>
                  <span className={`text-sm font-extrabold ${selectedAlert.deviation >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {selectedAlert.deviation >= 0 ? '+' : ''}{selectedAlert.deviation_percent.toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* What Happened Section */}
              <div className="space-y-2">
                <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                  What Happened
                </h4>
                <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 text-xs text-slate-300 leading-relaxed">
                  {selectedAlert.explanation || selectedAlert.evidence_snapshot?.explanation || 'Statistical anomaly detected against rolling historical baseline.'}
                </div>
              </div>

              {/* Evidence & Top Contributors */}
              {selectedAlert.top_drivers && selectedAlert.top_drivers.length > 0 && (
                <div className="space-y-2">
                  <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                    Key Empirical Drivers
                  </h4>
                  <div className="space-y-2">
                    {selectedAlert.top_drivers.map((d, i) => (
                      <div
                        key={i}
                        className="p-3 bg-slate-950/50 border border-slate-800 rounded-xl text-xs flex items-center justify-between"
                      >
                        <div>
                          <span className="font-semibold text-slate-200">{d.driver_name}</span>
                          <span className="text-slate-500 block text-[10px] capitalize">
                            Type: {d.driver_type} • Confidence: {d.confidence}
                          </span>
                        </div>
                        <span className="font-bold text-indigo-300">
                          {d.contribution_score}% impact
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Connected Workflows */}
              <div className="space-y-2.5 pt-2">
                <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                  Connected Intelligence Workflows
                </h4>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <button
                    type="button"
                    onClick={() => {
                      setInvestigationAnomalyId(selectedAlert.anomaly_id);
                      setSelectedAlert(null);
                    }}
                    className="p-3 rounded-xl bg-slate-950 border border-slate-800 hover:border-indigo-500/50 text-left group transition-all"
                  >
                    <div className="flex items-center space-x-2 text-indigo-400 font-semibold text-xs mb-1">
                      <Search className="w-3.5 h-3.5" />
                      <span>Root-Cause Investigation</span>
                    </div>
                    <p className="text-[11px] text-slate-400">
                      View multi-dimensional breakdown, promotions, and drift.
                    </p>
                  </button>

                  <button
                    type="button"
                    onClick={() => handleSimulate(selectedAlert)}
                    className="p-3 rounded-xl bg-slate-950 border border-slate-800 hover:border-emerald-500/50 text-left group transition-all"
                  >
                    <div className="flex items-center space-x-2 text-emerald-400 font-semibold text-xs mb-1">
                      <Sliders className="w-3.5 h-3.5" />
                      <span>What-If Simulation</span>
                    </div>
                    <p className="text-[11px] text-slate-400">
                      Explore hypothetical scenarios over 7, 30, and 90 days.
                    </p>
                  </button>
                </div>
              </div>
            </div>

            {/* Bottom Actions */}
            <div className="pt-4 border-t border-slate-800 flex items-center justify-between">
              <div className="flex items-center space-x-2">
                {selectedAlert.status === 'new' && (
                  <button
                    type="button"
                    onClick={() => handleAcknowledge(selectedAlert.id)}
                    className="px-3 py-2 rounded-xl text-xs font-bold bg-amber-600 hover:bg-amber-500 text-white transition-all shadow-md"
                  >
                    Acknowledge Alert
                  </button>
                )}
                {(selectedAlert.status === 'new' || selectedAlert.status === 'acknowledged') && (
                  <>
                    <button
                      type="button"
                      onClick={() => handleResolve(selectedAlert.id)}
                      className="px-3 py-2 rounded-xl text-xs font-bold bg-emerald-600 hover:bg-emerald-500 text-white transition-all shadow-md"
                    >
                      Resolve Alert
                    </button>
                    <button
                      type="button"
                      onClick={() => handleDismiss(selectedAlert.id)}
                      className="px-3 py-2 rounded-xl text-xs font-bold bg-slate-800 hover:bg-slate-700 text-slate-300 transition-all"
                    >
                      Dismiss
                    </button>
                  </>
                )}
              </div>
              <button
                type="button"
                onClick={() => setSelectedAlert(null)}
                className="px-3 py-2 rounded-xl text-xs font-bold text-slate-400 hover:text-white"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Investigation Drawer Integration */}
      {investigationAnomalyId && (
        <InvestigationDrawer
          anomalyId={investigationAnomalyId}
          isOpen={Boolean(investigationAnomalyId)}
          onClose={() => setInvestigationAnomalyId(null)}
        />
      )}
    </div>
  );
};

export default IntelligenceMonitor;
