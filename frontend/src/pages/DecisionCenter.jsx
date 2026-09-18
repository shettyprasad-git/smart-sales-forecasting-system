import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  ShieldCheck,
  Clock,
  CheckCircle2,
  XCircle,
  RotateCcw,
  Layers,
  Sparkles,
  AlertCircle,
  FileText,
  ArrowRight,
  Filter,
} from 'lucide-react';
import { listDecisionsApi, getDecisionApi } from '../api/decisions';
import DecisionReviewPanel from '../components/DecisionReviewPanel';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import Modal from '../components/Modal';
import { formatDate } from '../utils/formatters';

const STATUS_FILTERS = [
  { id: 'all', label: 'All Decisions' },
  { id: 'pending_review', label: 'Pending Review' },
  { id: 'approved', label: 'Approved' },
  { id: 'rejected', label: 'Rejected' },
  { id: 'changes_requested', label: 'Changes Requested' },
];

const STATUS_BADGES = {
  pending_review: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
  approved: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
  rejected: 'bg-rose-500/15 text-rose-300 border-rose-500/30',
  changes_requested: 'bg-sky-500/15 text-sky-300 border-sky-500/30',
};

const DecisionCenter = () => {
  const [searchParams] = useSearchParams();
  const directId = searchParams.get('id');

  const [activeFilter, setActiveFilter] = useState('all');
  const [decisions, setDecisions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedDecision, setSelectedDecision] = useState(null);
  const [modalOpen, setModalOpen] = useState(false);

  const fetchDecisions = async () => {
    setLoading(true);
    setError(null);
    try {
      const params = {};
      if (activeFilter !== 'all') {
        params.status = activeFilter;
      }
      const data = await listDecisionsApi(params);
      setDecisions(data);

      if (directId) {
        const found = data.find((d) => d.id === directId);
        if (found) {
          setSelectedDecision(found);
          setModalOpen(true);
        } else {
          try {
            const directItem = await getDecisionApi(directId);
            setSelectedDecision(directItem);
            setModalOpen(true);
          } catch {
            // direct lookup failed
          }
        }
      }
    } catch (err) {
      console.error('Failed to load decisions:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to load decisions.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDecisions();
  }, [activeFilter]);

  const handleOpenReview = (decision) => {
    setSelectedDecision(decision);
    setModalOpen(true);
  };

  const handleDecisionUpdated = (updated) => {
    setSelectedDecision(updated);
    setDecisions((prev) => prev.map((d) => (d.id === updated.id ? updated : d)));
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center space-x-2 text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-1">
            <span>Governance & Decision Control</span>
            <span>•</span>
            <span className="text-slate-400">Phase 6.7</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-100 flex items-center space-x-3">
            <span>Executive Decision Center</span>
            <span className="text-xs px-2.5 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 font-mono font-medium">
              Human-in-the-Loop
            </span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Review recommendations, inspect empirical evidence, evaluate simulation outcomes, and record binding governance decisions.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <span className="px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-300 font-mono">
            {decisions.length} Decisions Logged
          </span>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex flex-wrap items-center gap-2 bg-slate-900/60 p-1.5 rounded-2xl border border-slate-800">
        {STATUS_FILTERS.map((f) => {
          const isActive = activeFilter === f.id;
          return (
            <button
              key={f.id}
              type="button"
              onClick={() => setActiveFilter(f.id)}
              className={`px-3.5 py-2 rounded-xl text-xs font-semibold transition-all cursor-pointer ${
                isActive
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-950/40'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              {f.label}
            </button>
          );
        })}
      </div>

      {/* Content Body */}
      {loading ? (
        <div className="py-16 flex justify-center">
          <LoadingSpinner />
        </div>
      ) : error ? (
        <ErrorMessage message={error} onRetry={fetchDecisions} />
      ) : decisions.length === 0 ? (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-12 text-center space-y-3">
          <div className="w-12 h-12 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center mx-auto">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <h3 className="text-base font-bold text-slate-200">No Decision Records Found</h3>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            {activeFilter === 'all'
              ? 'No recommendations have been submitted for review yet. Submit a recommendation from the Anomaly Investigation drawer to begin governance review.'
              : `There are currently no decision records in '${activeFilter.replace('_', ' ')}' state.`}
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {decisions.map((dec) => {
            const rec = dec.evidence_snapshot?.recommendation || {};
            const isApproved = dec.status === 'approved';
            const isRejected = dec.status === 'rejected';
            const isPending = dec.status === 'pending_review';

            return (
              <div
                key={dec.id}
                className="bg-slate-900 border border-slate-800/90 rounded-2xl p-5 shadow-lg hover:border-slate-700 transition-all flex flex-col justify-between space-y-4"
              >
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 font-mono">
                      {dec.recommendation_type.replace('_', ' ')}
                    </span>
                    <span
                      className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase border ${
                        STATUS_BADGES[dec.status] || STATUS_BADGES.pending_review
                      }`}
                    >
                      {dec.status.replace('_', ' ')}
                    </span>
                  </div>

                  <div>
                    <h3 className="text-sm font-bold text-slate-100 line-clamp-2 leading-snug">
                      {dec.modified_action || dec.proposed_action}
                    </h3>
                  </div>

                  {/* Context Badges */}
                  <div className="flex flex-wrap gap-1.5 pt-1 text-[10px]">
                    {dec.anomaly_id && (
                      <span className="px-2 py-0.5 rounded bg-slate-950/60 text-slate-300 border border-slate-800 font-mono">
                        {dec.anomaly_id}
                      </span>
                    )}
                    {dec.simulation_id && (
                      <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/20 font-mono">
                        Simulation Attached
                      </span>
                    )}
                    {rec.priority && (
                      <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 capitalize">
                        {rec.priority} Priority
                      </span>
                    )}
                  </div>
                </div>

                <div className="border-t border-slate-800/80 pt-3 flex items-center justify-between text-xs">
                  <span className="text-[11px] text-slate-400 font-mono">
                    {formatDate(dec.created_at)}
                  </span>
                  <button
                    type="button"
                    onClick={() => handleOpenReview(dec)}
                    className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition-colors cursor-pointer"
                  >
                    <span>{isPending ? 'Review & Decide' : 'View Record'}</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Review Modal */}
      {modalOpen && selectedDecision && (
        <Modal
          isOpen={modalOpen}
          onClose={() => setModalOpen(false)}
          title="Decision Governance Review"
          maxWidth="max-w-4xl"
        >
          <DecisionReviewPanel
            decision={selectedDecision}
            onDecisionUpdated={handleDecisionUpdated}
            onClose={() => setModalOpen(false)}
          />
        </Modal>
      )}
    </div>
  );
};

export default DecisionCenter;
