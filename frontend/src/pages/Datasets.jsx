import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Database,
  UploadCloud,
  FileSpreadsheet,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Trash2,
  RefreshCw,
  TrendingUp,
  ArrowRight,
  ShieldCheck,
  Info,
  Calendar,
  Layers,
  Package,
  Cpu,
} from 'lucide-react';
import {
  getCurrentDatasetApi,
  getDatasetHistoryApi,
  uploadDatasetApi,
  activateDatasetApi,
  deleteDatasetApi,
} from '../api/datasets';
import { getCurrentModelsApi, trainDatasetModelsApi } from '../api/models';
import { extractErrorMessage } from '../api/axios';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import Modal from '../components/Modal';
import { formatDate, formatNumber } from '../utils/formatters';

const STATUS_BADGES = {
  active: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
  archived: 'bg-slate-700/30 text-slate-400 border-slate-700/50',
  uploaded: 'bg-indigo-500/15 text-indigo-300 border-indigo-500/30',
  failed: 'bg-rose-500/15 text-rose-300 border-rose-500/30',
};

const Datasets = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);

  const [currentDataset, setCurrentDataset] = useState(null);
  const [history, setHistory] = useState([]);
  const [modelsData, setModelsData] = useState(null);
  const [retrainLoading, setRetrainLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);

  // Upload state
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isDragging, setIsDragging] = useState(false);

  // Delete modal state
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const [currentRes, historyRes, modelsRes] = await Promise.allSettled([
        getCurrentDatasetApi(),
        getDatasetHistoryApi(),
        getCurrentModelsApi(),
      ]);

      if (currentRes.status === 'fulfilled') {
        setCurrentDataset(currentRes.value);
      } else {
        setCurrentDataset(null);
      }

      if (historyRes.status === 'fulfilled') {
        setHistory(historyRes.value?.datasets || historyRes.value?.items || []);
      } else {
        setHistory([]);
      }

      if (modelsRes.status === 'fulfilled') {
        setModelsData(modelsRes.value);
      } else {
        setModelsData(null);
      }
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Auto-poll models data if any training job is active
  useEffect(() => {
    const isJobActive =
      modelsData?.training_job &&
      ['queued', 'processing', 'training', 'evaluating'].includes(modelsData.training_job.status);
    const areModelsActive =
      modelsData?.models &&
      modelsData.models.some((m) =>
        ['queued', 'processing', 'training', 'evaluating'].includes(m.status)
      );

    if (!isJobActive && !areModelsActive) return;

    const interval = setInterval(async () => {
      try {
        const updated = await getCurrentModelsApi();
        setModelsData(updated);
      } catch (e) {
        // silent polling catch
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [modelsData]);

  const handleRetrain = async () => {
    const targetId = currentDataset?.dataset_id || currentDataset?.id;
    if (!targetId) return;

    try {
      setRetrainLoading(true);
      setError(null);
      await trainDatasetModelsApi(targetId);
      setSuccessMessage('Model benchmarking and training queued for active dataset.');
      const updated = await getCurrentModelsApi();
      setModelsData(updated);
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setRetrainLoading(false);
    }
  };

  // Handle Drag & Drop
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      validateAndSetFile(files[0]);
    }
  };

  const handleFileChange = (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      validateAndSetFile(files[0]);
    }
  };

  const validateAndSetFile = (file) => {
    setError(null);
    setSuccessMessage(null);
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setError('Please select a valid CSV file (.csv).');
      setSelectedFile(null);
      return;
    }
    if (file.size > 50 * 1024 * 1024) {
      setError('File size exceeds 50MB limit.');
      setSelectedFile(null);
      return;
    }
    setSelectedFile(file);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    try {
      setActionLoading(true);
      setError(null);
      setSuccessMessage(null);
      setUploadProgress(10);

      const formData = new FormData();
      formData.append('file', selectedFile);

      const result = await uploadDatasetApi(formData, (progressEvent) => {
        if (progressEvent.total) {
          const percent = Math.round((progressEvent.loaded * 90) / progressEvent.total);
          setUploadProgress(percent);
        }
      });

      setUploadProgress(100);
      setSuccessMessage(
        `Dataset "${result.original_filename}" ingested successfully (${formatNumber(
          result.row_count
        )} rows) and activated!`
      );
      setSelectedFile(null);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
      await loadData();
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setActionLoading(false);
      setTimeout(() => setUploadProgress(0), 1000);
    }
  };

  const handleActivate = async (datasetId) => {
    try {
      setActionLoading(true);
      setError(null);
      setSuccessMessage(null);
      await activateDatasetApi(datasetId);
      setSuccessMessage('Dataset activated successfully! Real-time analytics have been updated.');
      await loadData();
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setActionLoading(false);
    }
  };

  const confirmDelete = (dataset) => {
    setDeleteTarget(dataset);
    setDeleteModalOpen(true);
  };

  const handleDelete = async () => {
    if (!deleteTarget) return;

    try {
      setActionLoading(true);
      setError(null);
      await deleteDatasetApi(deleteTarget.id);
      setDeleteModalOpen(false);
      setDeleteTarget(null);
      setSuccessMessage('Dataset and its sales records deleted.');
      await loadData();
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setActionLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Top Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800/80 pb-6">
        <div>
          <div className="flex items-center space-x-2 text-xs font-bold text-indigo-400 uppercase tracking-wider mb-1">
            <Database className="w-3.5 h-3.5" />
            <span>Runtime Data Pipeline</span>
          </div>
          <h1 className="text-2xl font-extrabold text-slate-100 tracking-tight">
            Dataset Management
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Upload CSV sales data to activate real-time forecasting, anomaly detection, root-cause investigations, and simulation models.
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={() => navigate('/forecast')}
            className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700/60 transition-all cursor-pointer"
          >
            <TrendingUp className="w-3.5 h-3.5 text-indigo-400" />
            <span>View Forecasts</span>
          </button>
          <button
            onClick={() => navigate('/dashboard')}
            className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-950/50 transition-all cursor-pointer"
          >
            <span>Executive Dashboard</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Alerts */}
      {error && (
        <ErrorMessage
          title="Pipeline Operation Failed"
          message={error}
          onRetry={loadData}
        />
      )}

      {successMessage && (
        <div className="p-4 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 flex items-start space-x-3 text-sm">
          <CheckCircle2 className="w-5 h-5 flex-shrink-0 text-emerald-400 mt-0.5" />
          <div className="flex-1 font-medium">{successMessage}</div>
        </div>
      )}

      {/* Active Dataset Overview Banner */}
      <div className="rounded-3xl bg-gradient-to-br from-slate-900 via-slate-900/90 to-slate-950 border border-slate-800 p-6 shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 rounded-xl bg-indigo-600/20 text-indigo-400 border border-indigo-500/30">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h2 className="text-base font-bold text-slate-100">
                  {currentDataset ? (currentDataset.filename || currentDataset.original_filename) : 'No Active Dataset'}
                </h2>
                <span
                  className={`px-2.5 py-0.5 rounded-full text-[10px] font-semibold border ${
                    currentDataset
                      ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
                      : 'bg-amber-500/15 text-amber-300 border-amber-500/30'
                  }`}
                >
                  {currentDataset ? 'Active Runtime Source' : 'Repository Default Fallback'}
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                {currentDataset
                  ? `Activated on ${formatDate(currentDataset.activated_at || currentDataset.created_at || new Date().toISOString())}`
                  : 'Upload a sales CSV below to switch the forecasting platform to your custom business dataset.'}
              </p>
            </div>
          </div>

          <button
            onClick={loadData}
            disabled={loading}
            className="p-2 rounded-xl bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700/60 transition-all cursor-pointer self-start sm:self-auto"
            title="Refresh dataset state"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin text-indigo-400' : ''}`} />
          </button>
        </div>

        {currentDataset ? (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-2">
            <div className="p-3.5 rounded-2xl bg-slate-950/60 border border-slate-800/80">
              <span className="text-[11px] font-medium text-slate-400 block">Total Records</span>
              <span className="text-lg font-bold text-slate-100 mt-0.5 block">
                {formatNumber(currentDataset.rows_imported ?? currentDataset.row_count ?? 0)}
              </span>
            </div>
            <div className="p-3.5 rounded-2xl bg-slate-950/60 border border-slate-800/80">
              <span className="text-[11px] font-medium text-slate-400 block">Unique Products</span>
              <span className="text-lg font-bold text-slate-100 mt-0.5 block">
                {formatNumber(currentDataset.products ?? currentDataset.product_count ?? 0)}
              </span>
            </div>
            <div className="p-3.5 rounded-2xl bg-slate-950/60 border border-slate-800/80">
              <span className="text-[11px] font-medium text-slate-400 block">Categories</span>
              <span className="text-lg font-bold text-slate-100 mt-0.5 block">
                {formatNumber(currentDataset.categories ?? currentDataset.category_count ?? 0)}
              </span>
            </div>
            <div className="p-3.5 rounded-2xl bg-slate-950/60 border border-slate-800/80">
              <span className="text-[11px] font-medium text-slate-400 block">Date Range</span>
              <span className="text-xs font-semibold text-slate-200 mt-1 block truncate">
                {currentDataset.start_date || currentDataset.min_date} → {currentDataset.end_date || currentDataset.max_date}
              </span>
            </div>
          </div>
        ) : (
          <div className="p-4 rounded-2xl bg-amber-500/10 border border-amber-500/20 text-amber-300/90 text-xs flex items-center space-x-3">
            <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0" />
            <span>
              In development mode, queries fall back to repository CSVs. In production mode, an uploaded active dataset is required.
            </span>
          </div>
        )}

        {/* Model Inference Transparency Notice */}
        <div className="p-3.5 rounded-2xl bg-indigo-950/40 border border-indigo-500/20 text-indigo-300 text-xs flex items-center justify-between">
          <div className="flex items-center space-x-2.5">
            <ShieldCheck className="w-4 h-4 text-indigo-400 flex-shrink-0" />
            <span>
              <strong className="text-indigo-200">Model Runtime Inference Mode:</strong> Custom company-specific models are benchmarked per horizon and served dynamically, with automatic fallback to global pre-trained models.
            </span>
          </div>
          <span className="hidden sm:inline-block text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 whitespace-nowrap ml-2">
            Dynamic Tenant Inference
          </span>
        </div>
      </div>

      {/* Company Forecasting Models Panel */}
      <div className="rounded-3xl bg-slate-900 border border-slate-800 p-6 shadow-xl space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800/80 pb-4">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 rounded-xl bg-purple-600/20 text-purple-400 border border-purple-500/30">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h2 className="text-base font-bold text-slate-100">Company Forecasting Models</h2>
                {modelsData?.active_model_version && (
                  <span className="px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-purple-500/15 text-purple-300 border border-purple-500/30">
                    Version {modelsData.active_model_version}
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Automated one-time benchmarking and selection per horizon (7D, 30D, 90D) evaluated on Validation WAPE.
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={handleRetrain}
              disabled={
                retrainLoading ||
                !currentDataset ||
                Boolean(
                  modelsData?.training_job &&
                    ['queued', 'processing', 'training', 'evaluating'].includes(
                      modelsData.training_job.status
                    )
                )
              }
              className="px-3.5 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs shadow-lg shadow-indigo-600/20 transition-all flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
            >
              <RefreshCw
                className={`w-3.5 h-3.5 ${
                  retrainLoading ||
                  (modelsData?.training_job &&
                    ['queued', 'processing', 'training', 'evaluating'].includes(
                      modelsData.training_job.status
                    ))
                    ? 'animate-spin'
                    : ''
                }`}
              />
              <span>{retrainLoading ? 'Triggering...' : 'Retrain Models'}</span>
            </button>
          </div>
        </div>

        {/* Training In Progress Banner */}
        {modelsData?.training_job &&
          ['queued', 'processing', 'training', 'evaluating'].includes(
            modelsData.training_job.status
          ) && (
            <div className="p-4 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-300 space-y-2">
              <div className="flex items-center justify-between text-xs font-semibold">
                <div className="flex items-center space-x-2">
                  <RefreshCw className="w-4 h-4 animate-spin text-amber-400" />
                  <span>
                    Training Job in Progress:{' '}
                    {modelsData.training_job.progress_stage ||
                      'Benchmarking candidate models...'}
                  </span>
                </div>
                <span className="uppercase text-[10px] tracking-wider px-2 py-0.5 rounded bg-amber-500/20 border border-amber-500/30">
                  {modelsData.training_job.status}
                </span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                <div className="bg-amber-400 h-1.5 rounded-full animate-pulse w-3/4"></div>
              </div>
            </div>
          )}

        {/* 3 Horizon Model Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-1">
          {(
            modelsData?.models || [
              { horizon: 7, status: 'none' },
              { horizon: 30, status: 'none' },
              { horizon: 90, status: 'none' },
            ]
          ).map((m) => {
            const isReady = m.status === 'ready';
            const isTraining = ['queued', 'processing', 'training', 'evaluating'].includes(
              m.status
            );
            const isInsufficient = m.status === 'insufficient_data';
            const isFailed = m.status === 'failed';

            return (
              <div
                key={m.horizon}
                className={`p-4 rounded-2xl border transition-all space-y-3 ${
                  isReady
                    ? 'bg-slate-950/70 border-emerald-500/30'
                    : isInsufficient
                    ? 'bg-slate-950/60 border-indigo-500/20'
                    : isTraining
                    ? 'bg-slate-950/60 border-amber-500/30'
                    : 'bg-slate-950/40 border-slate-800'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-200">
                    {m.horizon}-Day Forecast Model
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded-full text-[10px] font-semibold border ${
                      isReady
                        ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
                        : isTraining
                        ? 'bg-amber-500/15 text-amber-300 border-amber-500/30'
                        : isInsufficient
                        ? 'bg-indigo-500/15 text-indigo-300 border-indigo-500/30'
                        : isFailed
                        ? 'bg-rose-500/15 text-rose-300 border-rose-500/30'
                        : 'bg-slate-800 text-slate-400 border-slate-700'
                    }`}
                  >
                    {isReady
                      ? 'Ready'
                      : isTraining
                      ? 'Training...'
                      : isInsufficient
                      ? 'Insufficient History'
                      : isFailed
                      ? 'Failed'
                      : 'Not Trained'}
                  </span>
                </div>

                <div>
                  <span className="text-[11px] font-medium text-slate-400 block">
                    Selected model for this dataset
                  </span>
                  <span className="text-sm font-bold text-slate-100 mt-0.5 block truncate">
                    {isReady
                      ? m.model_type
                      : isInsufficient
                      ? 'Global Pretrained Fallback'
                      : 'Global Fallback'}
                  </span>
                </div>

                {isReady ? (
                  <div className="space-y-2 pt-1 border-t border-slate-800/80">
                    <div className="grid grid-cols-2 gap-2 text-[11px]">
                      <div className="p-2 rounded-xl bg-slate-900/80 border border-slate-800">
                        <span className="text-slate-400 block">Val WAPE</span>
                        <span className="font-bold text-emerald-400 mt-0.5 block">
                          {m.validation_wape != null
                            ? `${(m.validation_wape * 100).toFixed(1)}%`
                            : '—'}
                        </span>
                      </div>
                      <div className="p-2 rounded-xl bg-slate-900/80 border border-slate-800">
                        <span className="text-slate-400 block">Test WAPE</span>
                        <span className="font-bold text-slate-200 mt-0.5 block">
                          {m.test_wape != null
                            ? `${(m.test_wape * 100).toFixed(1)}%`
                            : '—'}
                        </span>
                      </div>
                    </div>
                    <div className="text-[10px] text-slate-400 flex items-center justify-between px-1">
                      <span>Trained rows: {m.training_rows || '—'}</span>
                      <span>{m.trained_at ? formatDate(m.trained_at) : ''}</span>
                    </div>
                  </div>
                ) : (
                  <div className="p-2.5 rounded-xl bg-slate-900/60 border border-slate-800/80 text-[11px] text-slate-400">
                    {m.status_message ||
                      (isTraining
                        ? 'Evaluating candidate algorithms...'
                        : 'Runtime queries use global pre-trained models.')}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Upload Section & CSV Guidelines */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Upload Dropzone */}
        <div className="lg:col-span-2 rounded-3xl bg-slate-900/90 border border-slate-800 p-6 shadow-xl flex flex-col justify-between space-y-5">
          <div>
            <div className="flex items-center space-x-2 text-xs font-bold text-indigo-400 uppercase tracking-wider">
              <UploadCloud className="w-4 h-4" />
              <span>Ingest New Dataset</span>
            </div>
            <h3 className="text-base font-bold text-slate-100 mt-1">
              Upload Sales Transactions (.csv)
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Drag & drop your CSV file here or browse from your computer. Max 50MB.
            </p>
          </div>

          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 flex flex-col items-center justify-center space-y-3 ${
              isDragging
                ? 'border-indigo-500 bg-indigo-500/10'
                : selectedFile
                ? 'border-emerald-500/60 bg-emerald-500/5'
                : 'border-slate-700/80 hover:border-slate-600 bg-slate-950/40 hover:bg-slate-950/70'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="hidden"
            />
            <div className="p-4 rounded-full bg-indigo-600/15 text-indigo-400">
              <FileSpreadsheet className="w-8 h-8" />
            </div>

            {selectedFile ? (
              <div className="space-y-1">
                <div className="text-sm font-bold text-slate-100 flex items-center justify-center space-x-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  <span>{selectedFile.name}</span>
                </div>
                <div className="text-xs text-slate-400">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • Ready to validate and ingest
                </div>
              </div>
            ) : (
              <div className="space-y-1">
                <p className="text-xs font-semibold text-slate-200">
                  Drop your CSV file here, or{' '}
                  <span className="text-indigo-400 underline underline-offset-2">browse</span>
                </p>
                <p className="text-[11px] text-slate-500">Supports standard comma-separated sales records</p>
              </div>
            )}
          </div>

          {/* Progress Bar */}
          {uploadProgress > 0 && (
            <div className="space-y-1.5">
              <div className="flex justify-between text-[11px] text-slate-400 font-medium">
                <span>Uploading & Processing Data...</span>
                <span>{uploadProgress}%</span>
              </div>
              <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-indigo-500 to-emerald-400 transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center justify-between pt-2">
            {selectedFile ? (
              <button
                onClick={() => {
                  setSelectedFile(null);
                  if (fileInputRef.current) fileInputRef.current.value = '';
                }}
                className="text-xs font-semibold text-rose-400 hover:text-rose-300 cursor-pointer"
              >
                Clear selection
              </button>
            ) : (
              <span className="text-xs text-slate-500">No file chosen</span>
            )}

            <button
              onClick={handleUpload}
              disabled={!selectedFile || actionLoading}
              className={`px-5 py-2.5 rounded-xl text-xs font-semibold flex items-center space-x-2 transition-all cursor-pointer ${
                !selectedFile || actionLoading
                  ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/50'
                  : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-950/50'
              }`}
            >
              {actionLoading ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Ingesting Dataset...</span>
                </>
              ) : (
                <>
                  <UploadCloud className="w-3.5 h-3.5" />
                  <span>Upload & Activate</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Requirements & Guidelines */}
        <div className="rounded-3xl bg-slate-900/90 border border-slate-800 p-6 shadow-xl space-y-4">
          <div className="flex items-center space-x-2 text-xs font-bold text-indigo-400 uppercase tracking-wider">
            <Info className="w-4 h-4" />
            <span>Data Ingestion Rules</span>
          </div>

          <div className="space-y-3 text-xs text-slate-300">
            <div>
              <span className="font-semibold text-slate-200 block mb-1">Required Columns:</span>
              <ul className="list-disc list-inside space-y-0.5 text-slate-400">
                <li><code className="text-indigo-300 bg-slate-950 px-1 py-0.5 rounded">date</code> (YYYY-MM-DD)</li>
                <li><code className="text-indigo-300 bg-slate-950 px-1 py-0.5 rounded">product_id</code> (String / SKU)</li>
                <li><code className="text-indigo-300 bg-slate-950 px-1 py-0.5 rounded">quantity</code> (&gt; 0)</li>
                <li><code className="text-indigo-300 bg-slate-950 px-1 py-0.5 rounded">unit_price</code> (&ge; 0)</li>
              </ul>
            </div>

            <div>
              <span className="font-semibold text-slate-200 block mb-1">Optional Columns:</span>
              <ul className="list-disc list-inside space-y-0.5 text-slate-400">
                <li><code className="text-slate-300 bg-slate-950 px-1 py-0.5 rounded">product_name</code></li>
                <li><code className="text-slate-300 bg-slate-950 px-1 py-0.5 rounded">category</code></li>
                <li><code className="text-slate-300 bg-slate-950 px-1 py-0.5 rounded">discount_percent</code> (0–100)</li>
                <li><code className="text-slate-300 bg-slate-950 px-1 py-0.5 rounded">is_promotion</code>, <code className="text-slate-300 bg-slate-950 px-1 py-0.5 rounded">is_holiday</code></li>
                <li><code className="text-slate-300 bg-slate-950 px-1 py-0.5 rounded">sales_amount</code>, <code className="text-slate-300 bg-slate-950 px-1 py-0.5 rounded">profit</code></li>
              </ul>
            </div>

            <div className="pt-2 border-t border-slate-800 space-y-1 text-[11px] text-slate-400">
              <p className="flex items-center space-x-1.5">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                <span>Single active dataset per tenant enforced.</span>
              </p>
              <p className="flex items-center space-x-1.5">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                <span>Atomic database transaction (zero partial imports).</span>
              </p>
              <p className="flex items-center space-x-1.5">
                <ShieldCheck className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                <span>Requires &ge; 28 daily observations for forecasting.</span>
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Dataset History Table */}
      <div className="rounded-3xl bg-slate-900/90 border border-slate-800 p-6 shadow-xl space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
          <div className="flex items-center space-x-2 text-xs font-bold text-indigo-400 uppercase tracking-wider">
            <Clock className="w-4 h-4" />
            <span>Dataset Upload History ({history.length})</span>
          </div>
        </div>

        {history.length === 0 ? (
          <div className="text-center py-10 text-slate-400 text-xs">
            No datasets uploaded yet. Upload a CSV file to see history.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-950/60 text-[11px] font-semibold text-slate-400 uppercase tracking-wider border-b border-slate-800">
                <tr>
                  <th className="py-3 px-4">Dataset Name</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4">Rows</th>
                  <th className="py-3 px-4">Products</th>
                  <th className="py-3 px-4">Categories</th>
                  <th className="py-3 px-4">Date Range</th>
                  <th className="py-3 px-4">Uploaded</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {history.map((item) => {
                  const isActive = item.status === 'active';
                  return (
                    <tr key={item.id} className="hover:bg-slate-800/30 transition-colors">
                      <td className="py-3.5 px-4 font-semibold text-slate-100 flex items-center space-x-2">
                        <FileSpreadsheet className="w-4 h-4 text-indigo-400 flex-shrink-0" />
                        <span className="truncate max-w-[200px]">{item.original_filename}</span>
                      </td>
                      <td className="py-3.5 px-4">
                        <span
                          className={`px-2 py-0.5 rounded-full text-[10px] font-semibold border ${
                            STATUS_BADGES[item.status] || STATUS_BADGES.uploaded
                          }`}
                        >
                          {item.status.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-3.5 px-4">{formatNumber(item.row_count)}</td>
                      <td className="py-3.5 px-4">{formatNumber(item.product_count)}</td>
                      <td className="py-3.5 px-4">{formatNumber(item.category_count)}</td>
                      <td className="py-3.5 px-4 text-slate-400 whitespace-nowrap">
                        {item.min_date} → {item.max_date}
                      </td>
                      <td className="py-3.5 px-4 text-slate-400 whitespace-nowrap">
                        {formatDate(item.created_at)}
                      </td>
                      <td className="py-3.5 px-4 text-right space-x-2 whitespace-nowrap">
                        {!isActive && item.status !== 'failed' && (
                          <button
                            onClick={() => handleActivate(item.id)}
                            disabled={actionLoading}
                            className="px-3 py-1 rounded-lg text-xs font-semibold bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/30 transition-all cursor-pointer"
                          >
                            Activate
                          </button>
                        )}
                        <button
                          onClick={() => confirmDelete(item)}
                          disabled={actionLoading}
                          className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-rose-950/30 transition-all cursor-pointer"
                          title="Delete dataset"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      <Modal
        isOpen={deleteModalOpen}
        onClose={() => setDeleteModalOpen(false)}
        title="Confirm Dataset Deletion"
      >
        <div className="space-y-4">
          <div className="flex items-start space-x-3 text-slate-300 text-xs">
            <AlertTriangle className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
            <p>
              Are you sure you want to permanently delete{' '}
              <strong className="text-slate-100">{deleteTarget?.original_filename}</strong>? All associated
              sales records ({formatNumber(deleteTarget?.row_count || 0)} rows) will be permanently deleted from Supabase PostgreSQL.
            </p>
          </div>

          <div className="flex justify-end space-x-3 pt-4 border-t border-slate-800">
            <button
              onClick={() => setDeleteModalOpen(false)}
              className="px-4 py-2 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 transition-all cursor-pointer"
            >
              Cancel
            </button>
            <button
              onClick={handleDelete}
              disabled={actionLoading}
              className="px-4 py-2 rounded-xl text-xs font-semibold bg-rose-600 hover:bg-rose-500 text-white transition-all cursor-pointer"
            >
              {actionLoading ? 'Deleting...' : 'Delete Dataset'}
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
};

export default Datasets;
