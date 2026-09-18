import React, { useState } from 'react';
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  RotateCcw,
  Clock,
  ShieldCheck,
  FileText,
  Layers,
  Sparkles,
  IndianRupee,
  Calendar,
  User,
  Info,
  ArrowRight,
  Send,
} from 'lucide-react';
import {
  approveDecisionApi,
  rejectDecisionApi,
  requestChangesApi,
  resubmitDecisionApi,
} from '../api/decisions';
import { formatCurrency, formatQuantity, formatDate, formatPercent } from '../utils/formatters';

const STATUS_BADGES = {
  pending_review: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
  approved: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
  rejected: 'bg-rose-500/15 text-rose-300 border-rose-500/30',
  changes_requested: 'bg-sky-500/15 text-sky-300 border-sky-500/30',
};

const DecisionReviewPanel = ({ decision, onDecisionUpdated, onClose }) => {
  const [modifiedAction, setModifiedAction] = useState(
    decision.modified_action || decision.proposed_action || ''
  );
  const [decisionNote, setDecisionNote] = useState(decision.decision_note || '');
  const [actionType, setActionType] = useState(null); // 'approve' | 'reject' | 'request_changes' | 'resubmit'
  const [showConfirm, setShowConfirm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const isFinalized = decision.status === 'approved' || decision.status === 'rejected';
  const isPending = decision.status === 'pending_review';
  const isChangesRequested = decision.status === 'changes_requested';

  const handleActionClick = (type) => {
    setError(null);
    if ((type === 'reject' || type === 'request_changes') && !decisionNote.trim()) {
      setError('A rationale note is required before rejecting or requesting changes.');
      return;
    }
    setActionType(type);
    setShowConfirm(true);
  };

  const handleConfirmSubmit = async () => {
    setLoading(true);
    setError(null);
    try {
      let updated;
      if (actionType === 'approve') {
        updated = await approveDecisionApi(decision.id, {
          decision_note: decisionNote.trim() || undefined,
          modified_action: modifiedAction.trim() !== decision.proposed_action ? modifiedAction.trim() : undefined,
        });
      } else if (actionType === 'reject') {
        updated = await rejectDecisionApi(decision.id, {
          decision_note: decisionNote.trim(),
        });
      } else if (actionType === 'request_changes') {
        updated = await requestChangesApi(decision.id, {
          decision_note: decisionNote.trim(),
          modified_action: modifiedAction.trim() || undefined,
        });
      } else if (actionType === 'resubmit') {
        updated = await resubmitDecisionApi(decision.id, {
          modified_action: modifiedAction.trim() || undefined,
          decision_note: decisionNote.trim() || undefined,
        });
      }
      setShowConfirm(false);
      if (onDecisionUpdated) {
        onDecisionUpdated(updated);
      }
    } catch (err) {
      console.error('Decision action failed:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to submit decision.');
    } finally {
      setLoading(false);
    }
  };

  const recData = decision.evidence_snapshot?.recommendation || {};
  const anomData = decision.evidence_snapshot?.anomaly_context || {};
  const simData = decision.simulation_snapshot || decision.evidence_snapshot?.simulation;

  return (
    <div className="space-y-6">
      {/* Header with Status Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <div className="flex items-center space-x-2 text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
            <span>Governance Decision Record</span>
            <span>•</span>
            <span className="font-mono text-indigo-400">{decision.id}</span>
          </div>
          <h2 className="text-xl font-bold text-slate-100 flex items-center space-x-3">
            <span>{decision.recommendation_type.replace('_', ' ').toUpperCase()}</span>
            <span
              className={`text-xs px-2.5 py-0.5 rounded-full font-mono uppercase font-bold border ${
                STATUS_BADGES[decision.status] || STATUS_BADGES.pending_review
              }`}
            >
              {decision.status.replace('_', ' ')}
            </span>
          </h2>
        </div>

        <div className="text-xs text-slate-400 space-y-0.5 sm:text-right">
          <div>
            Created: <span className="text-slate-200">{formatDate(decision.created_at)}</span>
          </div>
          {decision.reviewed_at && (
            <div>
              Decided: <span className="text-slate-200">{formatDate(decision.reviewed_at)}</span>
            </div>
          )}
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="flex items-center space-x-3 p-3.5 rounded-xl bg-rose-950/40 border border-rose-800/50 text-rose-200 text-xs">
          <AlertTriangle className="w-5 h-5 text-rose-400 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Guardrail Callout */}
      <div className="p-3.5 rounded-xl bg-indigo-950/25 border border-indigo-800/40 text-xs text-indigo-200 flex items-start space-x-3">
        <ShieldCheck className="w-5 h-5 text-indigo-400 flex-shrink-0 mt-0.5" />
        <div className="space-y-0.5">
          <span className="font-bold text-indigo-300">Human Governance Boundary</span>
          <p className="text-indigo-200/90 leading-relaxed">
            {decision.status === 'approved'
              ? 'Human approved — no automatic execution performed. Qualified personnel must manually enact operational decisions.'
              : decision.status === 'rejected'
              ? 'Rejected by human reviewer. The recommendation will not be pursued.'
              : decision.status === 'changes_requested'
              ? 'Changes requested — awaiting author revision before re-review.'
              : 'Human approval required before consideration or execution. Zero autonomous business actions are taken.'}
          </p>
        </div>
      </div>

      {/* Grid: Recommendation vs Human Modification */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Original Proposed Action */}
        <div className="bg-slate-950/60 border border-slate-800 rounded-2xl p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center space-x-2">
              <FileText className="w-4 h-4 text-indigo-400" />
              <span>Original Proposed Action</span>
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
              Immutable
            </span>
          </div>
          <p className="text-xs text-slate-200 leading-relaxed bg-slate-900/80 p-3 rounded-xl border border-slate-800 font-normal">
            {decision.proposed_action}
          </p>
          {recData.title && (
            <div className="text-[11px] text-slate-400">
              Title: <span className="text-slate-300 font-medium">{recData.title}</span>
            </div>
          )}
        </div>

        {/* Human Modified Action */}
        <div className="bg-slate-950/60 border border-slate-800 rounded-2xl p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center space-x-2">
              <User className="w-4 h-4 text-emerald-400" />
              <span>Reviewer Modified Action</span>
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono">
              Editable Wording
            </span>
          </div>
          {isFinalized ? (
            <p className="text-xs text-slate-200 leading-relaxed bg-slate-900/80 p-3 rounded-xl border border-slate-800 font-normal">
              {decision.modified_action || 'No reviewer modifications were made.'}
            </p>
          ) : (
            <textarea
              rows={3}
              value={modifiedAction}
              onChange={(e) => setModifiedAction(e.target.value)}
              placeholder="Refine or customize the recommendation action for human execution..."
              className="w-full bg-slate-900 border border-slate-800 rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 resize-none"
            />
          )}
        </div>
      </div>

      {/* Anomaly & Simulation Context Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Empirical Anomaly Evidence */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-2">
          <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center space-x-2 border-b border-slate-800 pb-2">
            <Layers className="w-4 h-4 text-indigo-400" />
            <span>Anchored Anomaly Evidence</span>
          </span>
          {decision.anomaly_id ? (
            <div className="space-y-2 text-xs">
              <div className="flex justify-between text-slate-400">
                <span>Anomaly ID:</span>
                <span className="font-mono text-slate-200">{decision.anomaly_id}</span>
              </div>
              {anomData.actual !== undefined && (
                <div className="flex justify-between text-slate-400">
                  <span>Actual vs Expected:</span>
                  <span className="font-mono text-slate-200">
                    {formatQuantity(anomData.actual)} vs {formatQuantity(anomData.baseline)}
                  </span>
                </div>
              )}
              {anomData.deviation_percent !== undefined && (
                <div className="flex justify-between text-slate-400">
                  <span>Deviation:</span>
                  <span className="font-mono font-bold text-rose-400">
                    {formatPercent(anomData.deviation_percent)}
                  </span>
                </div>
              )}
              {recData.supporting_evidence?.length > 0 && (
                <div className="pt-2 border-t border-slate-800/80 space-y-1">
                  <span className="text-[11px] text-slate-400 font-semibold block">Supporting Drivers:</span>
                  <ul className="space-y-1 text-[11px] text-slate-300 list-disc list-inside">
                    {recData.supporting_evidence.map((ev, i) => (
                      <li key={i}>{ev}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <p className="text-xs text-slate-400 italic">No direct anomaly ID anchored.</p>
          )}
        </div>

        {/* Associated What-If Simulation */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-2">
          <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center space-x-2 border-b border-slate-800 pb-2">
            <Sparkles className="w-4 h-4 text-emerald-400" />
            <span>What-If Simulation Evidence</span>
          </span>
          {simData ? (
            <div className="space-y-2 text-xs">
              <div className="flex justify-between text-slate-400">
                <span>Scenario:</span>
                <span className="font-semibold text-slate-200">
                  {simData.scenario_type?.replace('_', ' ').toUpperCase()} ({simData.horizon_days}d)
                </span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Volume Variance:</span>
                <span className="font-mono font-bold text-emerald-400">
                  {simData.quantity_delta > 0 ? `+${formatQuantity(simData.quantity_delta)}` : formatQuantity(simData.quantity_delta)}
                </span>
              </div>
              {simData.revenue_delta !== null && simData.revenue_delta !== undefined && (
                <div className="flex justify-between text-slate-400">
                  <span>Revenue Variance:</span>
                  <span className="font-mono text-sky-400">
                    {simData.revenue_delta > 0 ? `+${formatCurrency(simData.revenue_delta)}` : formatCurrency(simData.revenue_delta)}
                  </span>
                </div>
              )}
              {simData.assumptions?.length > 0 && (
                <div className="pt-2 border-t border-slate-800/80 space-y-1">
                  <span className="text-[11px] text-slate-400 font-semibold block">Simulation Assumptions:</span>
                  <ul className="space-y-1 text-[11px] text-slate-300 list-disc list-inside">
                    {simData.assumptions.slice(0, 2).map((a, i) => (
                      <li key={i}>{a}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <p className="text-xs text-slate-400 italic">No what-if simulation attached to this decision.</p>
          )}
        </div>
      </div>

      {/* Reviewer Note / Rationale Input (if active) */}
      {!isFinalized && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-2">
          <label className="text-xs font-bold text-slate-300 uppercase tracking-wider block">
            Reviewer Decision Note / Rationale {isPending ? '(Required for Reject / Changes)' : ''}
          </label>
          <textarea
            rows={2}
            value={decisionNote}
            onChange={(e) => setDecisionNote(e.target.value)}
            placeholder="Record executive rationale, business context, or revision feedback..."
            className="w-full bg-slate-950/60 border border-slate-800 rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 resize-none"
          />
        </div>
      )}

      {/* Decision Note Display (if finalized) */}
      {isFinalized && decision.rationale && (
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 space-y-1.5">
          <span className="text-xs font-bold text-slate-400 uppercase tracking-wider block">
            Decision Rationale Recorded:
          </span>
          <p className="text-xs text-slate-200 font-medium italic bg-slate-950/60 p-3 rounded-xl border border-slate-800">
            "{decision.rationale}"
          </p>
        </div>
      )}

      {/* Decision Action Buttons */}
      {!isFinalized && (
        <div className="flex flex-wrap items-center justify-end gap-3 pt-2">
          {isPending && (
            <>
              <button
                type="button"
                onClick={() => handleActionClick('reject')}
                className="inline-flex items-center space-x-1.5 px-4 py-2 rounded-xl bg-rose-950/40 hover:bg-rose-900/60 text-rose-300 border border-rose-800/60 text-xs font-bold transition-all cursor-pointer"
              >
                <XCircle className="w-4 h-4" />
                <span>Reject</span>
              </button>

              <button
                type="button"
                onClick={() => handleActionClick('request_changes')}
                className="inline-flex items-center space-x-1.5 px-4 py-2 rounded-xl bg-sky-950/40 hover:bg-sky-900/60 text-sky-300 border border-sky-800/60 text-xs font-bold transition-all cursor-pointer"
              >
                <RotateCcw className="w-4 h-4" />
                <span>Request Changes</span>
              </button>

              <button
                type="button"
                onClick={() => handleActionClick('approve')}
                className="inline-flex items-center space-x-1.5 px-5 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold shadow-lg shadow-emerald-950/40 transition-all cursor-pointer"
              >
                <CheckCircle2 className="w-4 h-4" />
                <span>Approve for Human Execution</span>
              </button>
            </>
          )}

          {isChangesRequested && (
            <button
              type="button"
              onClick={() => handleActionClick('resubmit')}
              className="inline-flex items-center space-x-1.5 px-5 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold shadow-lg shadow-indigo-950/40 transition-all cursor-pointer"
            >
              <Send className="w-4 h-4" />
              <span>Resubmit for Review</span>
            </button>
          )}
        </div>
      )}

      {/* Confirmation Modal */}
      {showConfirm && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 max-w-md w-full shadow-2xl space-y-4">
            <div className="flex items-center space-x-3">
              <div
                className={`p-2.5 rounded-xl border ${
                  actionType === 'approve'
                    ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                    : actionType === 'reject'
                    ? 'bg-rose-500/10 border-rose-500/20 text-rose-400'
                    : 'bg-sky-500/10 border-sky-500/20 text-sky-400'
                }`}
              >
                {actionType === 'approve' ? (
                  <CheckCircle2 className="w-5 h-5" />
                ) : actionType === 'reject' ? (
                  <XCircle className="w-5 h-5" />
                ) : (
                  <RotateCcw className="w-5 h-5" />
                )}
              </div>
              <h3 className="text-base font-bold text-slate-100 capitalize">
                Confirm {actionType.replace('_', ' ')}
              </h3>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              {actionType === 'approve' &&
                'Approval records your executive decision and does NOT automatically execute any business action, order, or price adjustment. Are you sure you want to approve this recommendation?'}
              {actionType === 'reject' &&
                'Reject this recommendation? A permanent audit event will be recorded with your rationale note.'}
              {actionType === 'request_changes' &&
                'Send this recommendation back for revision? The author will be notified to modify wording.'}
              {actionType === 'resubmit' &&
                'Resubmit this revised recommendation for review? Status will return to pending review.'}
            </p>

            <div className="flex items-center justify-end space-x-3 pt-2">
              <button
                type="button"
                onClick={() => setShowConfirm(false)}
                disabled={loading}
                className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-400 hover:text-slate-200 bg-slate-800 hover:bg-slate-700 transition-colors cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleConfirmSubmit}
                disabled={loading}
                className={`px-4 py-2 rounded-xl text-xs font-bold text-white transition-all cursor-pointer ${
                  actionType === 'approve'
                    ? 'bg-emerald-600 hover:bg-emerald-500'
                    : actionType === 'reject'
                    ? 'bg-rose-600 hover:bg-rose-500'
                    : 'bg-sky-600 hover:bg-sky-500'
                }`}
              >
                {loading ? 'Recording Decision...' : 'Confirm Decision'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Chronological Audit Trail */}
      <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-lg space-y-3">
        <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
          <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center space-x-2">
            <Clock className="w-4 h-4 text-indigo-400" />
            <span>Immutable Governance Audit Trail</span>
          </span>
          <span className="text-[10px] text-slate-400 font-mono">
            {decision.audit_events?.length || 0} Events Recorded
          </span>
        </div>

        <div className="space-y-2.5 pt-1">
          {decision.audit_events?.map((evt, idx) => (
            <div
              key={evt.id || idx}
              className="flex items-start space-x-3 text-xs p-3 rounded-xl bg-slate-950/60 border border-slate-800/80"
            >
              <span
                className={`p-1.5 rounded-lg font-mono text-[10px] uppercase font-bold ${
                  evt.event_type === 'approved'
                    ? 'bg-emerald-500/10 text-emerald-400'
                    : evt.event_type === 'rejected'
                    ? 'bg-rose-500/10 text-rose-400'
                    : evt.event_type === 'changes_requested'
                    ? 'bg-sky-500/10 text-sky-400'
                    : 'bg-indigo-500/10 text-indigo-400'
                }`}
              >
                {evt.event_type}
              </span>
              <div className="flex-1 space-y-0.5">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-200">
                    Status:{' '}
                    {evt.previous_status ? `${evt.previous_status} → ${evt.new_status}` : evt.new_status}
                  </span>
                  <span className="text-[11px] text-slate-400 font-mono">
                    {formatDate(evt.created_at)}
                  </span>
                </div>
                {evt.note && <p className="text-slate-400 text-[11px] italic">"{evt.note}"</p>}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default DecisionReviewPanel;
