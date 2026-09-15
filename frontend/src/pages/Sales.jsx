import React, { useState, useEffect, useCallback } from 'react';
import { getSalesApi, createSaleApi, updateSaleApi, deleteSaleApi } from '../api/sales';
import { getProductsApi } from '../api/products';
import { extractErrorMessage } from '../api/axios';
import LoadingSpinner from '../components/LoadingSpinner';
import ErrorMessage from '../components/ErrorMessage';
import EmptyState from '../components/EmptyState';
import Modal from '../components/Modal';
import ConfirmDialog from '../components/ConfirmDialog';
import {
  Plus,
  Search,
  ShoppingCart,
  Edit2,
  Trash2,
  Calendar,
  Package,
  DollarSign,
  TrendingUp,
  Tag,
  Loader2,
  Check,
} from 'lucide-react';
import { formatCurrency, formatNumber, formatDate } from '../utils/formatters';

const Sales = () => {
  const [sales, setSales] = useState([]);
  const [products, setProducts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');

  // Modals
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);

  const [selectedSale, setSelectedSale] = useState(null);
  const [formError, setFormError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Form input state
  // CRITICAL: product_id here stores the INTEGER primary key (e.g. 1, 2, 3) of Product
  const [formData, setFormData] = useState({
    product_id: '',
    sale_date: new Date().toISOString().split('T')[0],
    quantity: '1',
    unit_price: '',
    discount_percent: '0',
    promotion: false,
    holiday_flag: false,
    sales_amount: '',
    profit: '0',
  });

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const [salesData, productsData] = await Promise.all([getSalesApi(), getProductsApi()]);
      setSales(salesData);
      setProducts(productsData);
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Product ID to Product Object map for fast table lookup
  const productMap = React.useMemo(() => {
    const map = new Map();
    products.forEach((p) => map.set(p.id, p));
    return map;
  }, [products]);

  // Auto calculate sales amount when quantity, unit_price, or discount_percent changes
  const calculateSalesAmount = (qtyStr, priceStr, discountStr) => {
    const qty = parseFloat(qtyStr) || 0;
    const price = parseFloat(priceStr) || 0;
    const discount = parseFloat(discountStr) || 0;
    const rawTotal = qty * price;
    const discountAmt = rawTotal * (discount / 100);
    return Math.max(0, rawTotal - discountAmt).toFixed(2);
  };

  const handleProductSelect = (selectedDbId) => {
    const dbId = parseInt(selectedDbId, 10);
    const prod = productMap.get(dbId);

    const newUnitPrice = prod ? prod.unit_price.toString() : formData.unit_price;
    const newSalesAmount = calculateSalesAmount(formData.quantity, newUnitPrice, formData.discount_percent);

    setFormData({
      ...formData,
      product_id: selectedDbId,
      unit_price: newUnitPrice,
      sales_amount: newSalesAmount,
    });
  };

  const handleFieldChange = (field, value) => {
    const updated = { ...formData, [field]: value };

    if (field === 'quantity' || field === 'unit_price' || field === 'discount_percent') {
      updated.sales_amount = calculateSalesAmount(
        updated.quantity,
        updated.unit_price,
        updated.discount_percent
      );
    }

    setFormData(updated);
  };

  const resetForm = () => {
    setFormData({
      product_id: products.length > 0 ? products[0].id.toString() : '',
      sale_date: new Date().toISOString().split('T')[0],
      quantity: '1',
      unit_price: products.length > 0 ? products[0].unit_price.toString() : '',
      discount_percent: '0',
      promotion: false,
      holiday_flag: false,
      sales_amount: products.length > 0 ? products[0].unit_price.toString() : '',
      profit: '0',
    });
    setFormError('');
    setSelectedSale(null);
  };

  const handleOpenAddModal = () => {
    resetForm();
    if (products.length > 0) {
      const defaultProd = products[0];
      setFormData({
        product_id: defaultProd.id.toString(),
        sale_date: new Date().toISOString().split('T')[0],
        quantity: '1',
        unit_price: defaultProd.unit_price.toString(),
        discount_percent: '0',
        promotion: false,
        holiday_flag: false,
        sales_amount: defaultProd.unit_price.toString(),
        profit: (defaultProd.unit_price * 0.2).toFixed(2), // 20% estimated default profit preview
      });
    }
    setIsAddModalOpen(true);
  };

  const handleOpenEditModal = (sale) => {
    setSelectedSale(sale);
    setFormData({
      product_id: sale.product_id.toString(),
      sale_date: sale.sale_date,
      quantity: sale.quantity.toString(),
      unit_price: sale.unit_price.toString(),
      discount_percent: sale.discount_percent.toString(),
      promotion: sale.promotion,
      holiday_flag: sale.holiday_flag,
      sales_amount: sale.sales_amount.toString(),
      profit: sale.profit.toString(),
    });
    setFormError('');
    setIsEditModalOpen(true);
  };

  const handleOpenDeleteDialog = (sale) => {
    setSelectedSale(sale);
    setIsDeleteOpen(true);
  };

  const validateSalePayload = () => {
    const prodIdInt = parseInt(formData.product_id, 10);
    if (isNaN(prodIdInt) || prodIdInt <= 0) {
      setFormError('Please select a valid product.');
      return null;
    }

    if (!formData.sale_date) {
      setFormError('Sale date is required.');
      return null;
    }

    const qty = parseFloat(formData.quantity);
    if (isNaN(qty) || qty <= 0) {
      setFormError('Quantity must be greater than 0.');
      return null;
    }

    const price = parseFloat(formData.unit_price);
    if (isNaN(price) || price < 0) {
      setFormError('Unit price must be a non-negative number.');
      return null;
    }

    const discount = parseFloat(formData.discount_percent);
    if (isNaN(discount) || discount < 0 || discount > 100) {
      setFormError('Discount percent must be between 0% and 100%.');
      return null;
    }

    const salesAmt = parseFloat(formData.sales_amount);
    if (isNaN(salesAmt) || salesAmt < 0) {
      setFormError('Sales amount must be a non-negative number.');
      return null;
    }

    const profitVal = parseFloat(formData.profit) || 0;

    return {
      product_id: prodIdInt,
      sale_date: formData.sale_date,
      quantity: qty,
      unit_price: price,
      discount_percent: discount,
      promotion: Boolean(formData.promotion),
      holiday_flag: Boolean(formData.holiday_flag),
      sales_amount: salesAmt,
      profit: profitVal,
    };
  };

  const handleCreateSubmit = async (e) => {
    e.preventDefault();
    setFormError('');

    const payload = validateSalePayload();
    if (!payload) return;

    try {
      setIsSubmitting(true);
      await createSaleApi(payload);
      setIsAddModalOpen(false);
      resetForm();
      fetchData();
    } catch (err) {
      setFormError(extractErrorMessage(err));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleUpdateSubmit = async (e) => {
    e.preventDefault();
    setFormError('');

    const payload = validateSalePayload();
    if (!payload) return;

    try {
      setIsSubmitting(true);
      await updateSaleApi(selectedSale.id, payload);
      setIsEditModalOpen(false);
      resetForm();
      fetchData();
    } catch (err) {
      setFormError(extractErrorMessage(err));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDeleteConfirm = async () => {
    if (!selectedSale) return;
    try {
      setIsSubmitting(true);
      await deleteSaleApi(selectedSale.id);
      setIsDeleteOpen(false);
      resetForm();
      fetchData();
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setIsSubmitting(false);
    }
  };

  // Filter sales
  const filteredSales = sales.filter((s) => {
    const term = searchTerm.toLowerCase();
    const prod = productMap.get(s.product_id);
    const prodName = prod ? prod.product_name.toLowerCase() : '';
    const prodExtId = prod ? prod.product_id.toLowerCase() : '';
    return (
      s.sale_date.includes(term) ||
      prodName.includes(term) ||
      prodExtId.includes(term) ||
      s.sales_amount.toString().includes(term)
    );
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800/80 pb-6">
        <div>
          <h1 className="text-2xl font-extrabold text-slate-100 tracking-tight">
            Sales Management
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Log transactions, track sales amounts, promotional discounts, and net profitability.
          </p>
        </div>

        <button
          onClick={handleOpenAddModal}
          disabled={products.length === 0}
          className="inline-flex items-center space-x-2 px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-950/50 transition-all cursor-pointer self-start sm:self-auto disabled:opacity-50"
        >
          <Plus className="w-4 h-4" />
          <span>Record New Sale</span>
        </button>
      </div>

      {products.length === 0 && !loading && (
        <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-500/30 text-amber-300 text-xs flex items-center justify-between">
          <span>No products found in catalog. Please create a product first before recording sales.</span>
        </div>
      )}

      {/* Search Filter */}
      <div className="relative max-w-md">
        <Search className="w-4 h-4 text-slate-500 absolute left-3.5 top-1/2 -translate-y-1/2" />
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          placeholder="Search by Date, Product Name, or Amount..."
          className="w-full pl-10 pr-4 py-2.5 bg-slate-900 border border-slate-800 rounded-xl text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all shadow-inner"
        />
      </div>

      {error && (
        <ErrorMessage
          title="Unable to load sales records"
          message={error}
          onRetry={fetchData}
        />
      )}

      {loading ? (
        <LoadingSpinner text="Fetching sales transaction history..." size="lg" />
      ) : filteredSales.length === 0 ? (
        <EmptyState
          icon={ShoppingCart}
          title={searchTerm ? 'No matching sales records' : 'No sales records'}
          description={
            searchTerm
              ? `No transactions matching "${searchTerm}" were found.`
              : 'Record your first sale to feed data to the forecasting ML engine.'
          }
          action={
            !searchTerm && products.length > 0 ? (
              <button
                onClick={handleOpenAddModal}
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-xl shadow-md cursor-pointer"
              >
                Record Sale
              </button>
            ) : null
          }
        />
      ) : (
        /* Sales Records Table */
        <div className="rounded-2xl border border-slate-800 bg-slate-900/80 shadow-xl overflow-hidden backdrop-blur-sm">
          <div className="overflow-x-auto custom-scrollbar">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-950/80 text-slate-400 font-semibold border-b border-slate-800">
                <tr>
                  <th className="py-3.5 px-4">Date</th>
                  <th className="py-3.5 px-4">Product Name</th>
                  <th className="py-3.5 px-4 text-right">Qty</th>
                  <th className="py-3.5 px-4 text-right">Unit Price</th>
                  <th className="py-3.5 px-4 text-center">Discount</th>
                  <th className="py-3.5 px-4 text-center">Flags</th>
                  <th className="py-3.5 px-4 text-right">Sales Amount</th>
                  <th className="py-3.5 px-4 text-right">Profit</th>
                  <th className="py-3.5 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {filteredSales.map((s) => {
                  const prod = productMap.get(s.product_id);
                  return (
                    <tr key={s.id} className="hover:bg-slate-800/40 transition-colors">
                      <td className="py-3.5 px-4 font-mono text-slate-300">
                        {formatDate(s.sale_date)}
                      </td>
                      <td className="py-3.5 px-4">
                        <div className="font-semibold text-slate-100">
                          {prod ? prod.product_name : `Product #${s.product_id}`}
                        </div>
                        {prod && (
                          <div className="text-[10px] font-mono text-indigo-400">
                            {prod.product_id}
                          </div>
                        )}
                      </td>
                      <td className="py-3.5 px-4 text-right font-mono font-medium text-slate-200">
                        {formatNumber(s.quantity)}
                      </td>
                      <td className="py-3.5 px-4 text-right font-mono text-slate-300">
                        {formatCurrency(s.unit_price)}
                      </td>
                      <td className="py-3.5 px-4 text-center font-mono text-slate-400">
                        {s.discount_percent > 0 ? (
                          <span className="px-2 py-0.5 rounded-md bg-amber-500/10 border border-amber-500/20 text-amber-400 text-[11px] font-semibold">
                            {s.discount_percent}%
                          </span>
                        ) : (
                          '0%'
                        )}
                      </td>
                      <td className="py-3.5 px-4 text-center space-x-1">
                        {s.promotion && (
                          <span className="px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 text-[10px] font-semibold">
                            PROMO
                          </span>
                        )}
                        {s.holiday_flag && (
                          <span className="px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-300 text-[10px] font-semibold">
                            HOLIDAY
                          </span>
                        )}
                      </td>
                      <td className="py-3.5 px-4 text-right font-mono font-bold text-indigo-300">
                        {formatCurrency(s.sales_amount)}
                      </td>
                      <td className="py-3.5 px-4 text-right font-mono font-bold text-emerald-400">
                        {formatCurrency(s.profit)}
                      </td>
                      <td className="py-3.5 px-4 text-right">
                        <div className="inline-flex items-center space-x-2">
                          <button
                            onClick={() => handleOpenEditModal(s)}
                            className="p-1.5 rounded-lg text-slate-400 hover:text-indigo-300 hover:bg-indigo-950/30 transition-colors cursor-pointer"
                            title="Edit Record"
                          >
                            <Edit2 className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => handleOpenDeleteDialog(s)}
                            className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-rose-950/30 transition-colors cursor-pointer"
                            title="Delete Record"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="px-4 py-3 bg-slate-950/40 border-t border-slate-800 text-[11px] text-slate-500 flex justify-between items-center">
            <span>Showing {filteredSales.length} of {sales.length} records</span>
          </div>
        </div>
      )}

      {/* Add Sale Modal */}
      <Modal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        title="Record New Sale"
      >
        <form onSubmit={handleCreateSubmit} className="space-y-4">
          {formError && (
            <div className="p-3 rounded-xl bg-rose-950/40 border border-rose-500/30 text-rose-300 text-xs">
              {formError}
            </div>
          )}

          {/* Product Select Dropdown: Maps Product Name to Database Integer ID */}
          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1">
              Select Product *
            </label>
            <div className="relative">
              <Package className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
              <select
                value={formData.product_id}
                onChange={(e) => handleProductSelect(e.target.value)}
                required
                className="w-full pl-9 pr-8 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-100 focus:outline-none focus:border-indigo-500 cursor-pointer"
              >
                {products.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.product_name} ({p.product_id}) — {formatCurrency(p.unit_price)}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Sale Date *
              </label>
              <input
                type="date"
                value={formData.sale_date}
                onChange={(e) => handleFieldChange('sale_date', e.target.value)}
                required
                className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-100 focus:outline-none focus:border-indigo-500 font-mono"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Quantity *
              </label>
              <input
                type="number"
                step="1"
                min="1"
                value={formData.quantity}
                onChange={(e) => handleFieldChange('quantity', e.target.value)}
                required
                className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-100 focus:outline-none focus:border-indigo-500 font-mono"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Unit Price (₹) *
              </label>
              <input
                type="number"
                step="0.01"
                min="0"
                value={formData.unit_price}
                onChange={(e) => handleFieldChange('unit_price', e.target.value)}
                required
                className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-100 focus:outline-none focus:border-indigo-500 font-mono"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Discount (%)
              </label>
              <input
                type="number"
                step="0.1"
                min="0"
                max="100"
                value={formData.discount_percent}
                onChange={(e) => handleFieldChange('discount_percent', e.target.value)}
                className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-100 focus:outline-none focus:border-indigo-500 font-mono"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Sales Amount (₹) *
              </label>
              <input
                type="number"
                step="0.01"
                min="0"
                value={formData.sales_amount}
                onChange={(e) => handleFieldChange('sales_amount', e.target.value)}
                required
                className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-100 focus:outline-none focus:border-indigo-500 font-mono font-bold"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Estimated Profit (₹)
              </label>
              <input
                type="number"
                step="0.01"
                value={formData.profit}
                onChange={(e) => handleFieldChange('profit', e.target.value)}
                className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-100 focus:outline-none focus:border-indigo-500 font-mono"
              />
            </div>
          </div>

          {/* Promotion & Holiday Flags */}
          <div className="flex items-center space-x-6 pt-1">
            <label className="flex items-center space-x-2 cursor-pointer select-none text-xs text-slate-300">
              <input
                type="checkbox"
                checked={formData.promotion}
                onChange={(e) => handleFieldChange('promotion', e.target.checked)}
                className="w-4 h-4 rounded border-slate-800 bg-slate-950 text-indigo-600 focus:ring-indigo-500"
              />
              <span>Promotional Sale</span>
            </label>

            <label className="flex items-center space-x-2 cursor-pointer select-none text-xs text-slate-300">
              <input
                type="checkbox"
                checked={formData.holiday_flag}
                onChange={(e) => handleFieldChange('holiday_flag', e.target.checked)}
                className="w-4 h-4 rounded border-slate-800 bg-slate-950 text-indigo-600 focus:ring-indigo-500"
              />
              <span>Holiday Period</span>
            </label>
          </div>

          <div className="flex items-center justify-end space-x-3 pt-4 border-t border-slate-800">
            <button
              type="button"
              onClick={() => setIsAddModalOpen(false)}
              className="px-4 py-2 text-xs font-medium text-slate-400 hover:text-slate-200 cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="inline-flex items-center space-x-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-xl cursor-pointer disabled:opacity-50"
            >
              {isSubmitting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
              <span>Save Record</span>
            </button>
          </div>
        </form>
      </Modal>

      {/* Edit Sale Modal */}
      <Modal
        isOpen={isEditModalOpen}
        onClose={() => setIsEditModalOpen(false)}
        title={`Edit Sale #${selectedSale?.id}`}
      >
        <form onSubmit={handleUpdateSubmit} className="space-y-4">
          {formError && (
            <div className="p-3 rounded-xl bg-rose-950/40 border border-rose-500/30 text-rose-300 text-xs">
              {formError}
            </div>
          )}

          <div>
            <label className="block text-xs font-semibold text-slate-300 mb-1">
              Select Product *
            </label>
            <select
              value={formData.product_id}
              onChange={(e) => handleProductSelect(e.target.value)}
              required
              className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-100 focus:outline-none focus:border-indigo-500 cursor-pointer"
            >
              {products.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.product_name} ({p.product_id})
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Sale Date *
              </label>
              <input
                type="date"
                value={formData.sale_date}
                onChange={(e) => handleFieldChange('sale_date', e.target.value)}
                required
                className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-100 focus:outline-none focus:border-indigo-500 font-mono"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Quantity *
              </label>
              <input
                type="number"
                step="1"
                min="1"
                value={formData.quantity}
                onChange={(e) => handleFieldChange('quantity', e.target.value)}
                required
                className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-100 focus:outline-none focus:border-indigo-500 font-mono"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Unit Price (₹) *
              </label>
              <input
                type="number"
                step="0.01"
                min="0"
                value={formData.unit_price}
                onChange={(e) => handleFieldChange('unit_price', e.target.value)}
                required
                className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-100 focus:outline-none focus:border-indigo-500 font-mono"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Discount (%)
              </label>
              <input
                type="number"
                step="0.1"
                min="0"
                max="100"
                value={formData.discount_percent}
                onChange={(e) => handleFieldChange('discount_percent', e.target.value)}
                className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-100 focus:outline-none focus:border-indigo-500 font-mono"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Sales Amount (₹) *
              </label>
              <input
                type="number"
                step="0.01"
                min="0"
                value={formData.sales_amount}
                onChange={(e) => handleFieldChange('sales_amount', e.target.value)}
                required
                className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-100 focus:outline-none focus:border-indigo-500 font-mono font-bold"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-300 mb-1">
                Profit (₹)
              </label>
              <input
                type="number"
                step="0.01"
                value={formData.profit}
                onChange={(e) => handleFieldChange('profit', e.target.value)}
                className="w-full px-3 py-2 bg-slate-950 border border-slate-800 rounded-xl text-xs text-slate-100 focus:outline-none focus:border-indigo-500 font-mono"
              />
            </div>
          </div>

          <div className="flex items-center space-x-6 pt-1">
            <label className="flex items-center space-x-2 cursor-pointer select-none text-xs text-slate-300">
              <input
                type="checkbox"
                checked={formData.promotion}
                onChange={(e) => handleFieldChange('promotion', e.target.checked)}
                className="w-4 h-4 rounded border-slate-800 bg-slate-950 text-indigo-600 focus:ring-indigo-500"
              />
              <span>Promotional Sale</span>
            </label>

            <label className="flex items-center space-x-2 cursor-pointer select-none text-xs text-slate-300">
              <input
                type="checkbox"
                checked={formData.holiday_flag}
                onChange={(e) => handleFieldChange('holiday_flag', e.target.checked)}
                className="w-4 h-4 rounded border-slate-800 bg-slate-950 text-indigo-600 focus:ring-indigo-500"
              />
              <span>Holiday Period</span>
            </label>
          </div>

          <div className="flex items-center justify-end space-x-3 pt-4 border-t border-slate-800">
            <button
              type="button"
              onClick={() => setIsEditModalOpen(false)}
              className="px-4 py-2 text-xs font-medium text-slate-400 hover:text-slate-200 cursor-pointer"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isSubmitting}
              className="inline-flex items-center space-x-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-xl cursor-pointer disabled:opacity-50"
            >
              {isSubmitting ? <Loader2 className="w-4 h-4 animate-spin" /> : null}
              <span>Save Changes</span>
            </button>
          </div>
        </form>
      </Modal>

      {/* Delete Dialog */}
      <ConfirmDialog
        isOpen={isDeleteOpen}
        onClose={() => setIsDeleteOpen(false)}
        onConfirm={handleDeleteConfirm}
        title={`Delete Sale Record #${selectedSale?.id}`}
        message={`Are you sure you want to delete this sale record for "${productMap.get(selectedSale?.product_id)?.product_name || 'Product'}" dated ${selectedSale?.sale_date}?`}
        loading={isSubmitting}
      />
    </div>
  );
};

export default Sales;
