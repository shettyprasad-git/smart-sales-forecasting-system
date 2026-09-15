import React from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';
import { formatCurrency, formatDate } from '../utils/formatters';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-900 border border-slate-700/80 p-3 rounded-xl shadow-xl text-xs space-y-1.5">
        <p className="font-semibold text-slate-300 border-b border-slate-800 pb-1">
          {formatDate(label)}
        </p>
        {payload.map((entry, index) => (
          <div key={`item-${index}`} className="flex items-center justify-between space-x-4">
            <span className="flex items-center space-x-1.5" style={{ color: entry.color }}>
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
              <span className="font-medium text-slate-300">{entry.name}:</span>
            </span>
            <span className="font-bold text-slate-100 font-mono">
              {formatCurrency(entry.value)}
            </span>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

const SalesChart = ({ data = [] }) => {
  if (!data || data.length === 0) {
    return (
      <div className="h-72 flex items-center justify-center text-slate-500 text-xs italic border border-slate-800/50 rounded-2xl bg-slate-900/30">
        No sales data available to render chart.
      </div>
    );
  }

  // Downsample large dataset for smooth interactive rendering if > 120 points
  const displayData = data.length > 120
    ? data.filter((_, idx) => idx % Math.ceil(data.length / 120) === 0)
    : data;

  return (
    <div className="w-full h-80 pt-2">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={displayData} margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
          <defs>
            <linearGradient id="colorSales" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#6366f1" stopOpacity={0.35} />
              <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
            </linearGradient>
            <linearGradient id="colorProfit" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.35} />
              <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />

          <XAxis
            dataKey="date"
            tickFormatter={(val) => {
              const d = new Date(val);
              return isNaN(d.getTime()) ? val : `${d.getDate()} ${d.toLocaleString('default', { month: 'short' })}`;
            }}
            stroke="#64748b"
            fontSize={11}
            tickLine={false}
            axisLine={false}
          />

          <YAxis
            stroke="#64748b"
            fontSize={11}
            tickLine={false}
            axisLine={false}
            tickFormatter={(val) => formatCurrency(val)}
            width={70}
          />

          <Tooltip content={<CustomTooltip />} />

          <Legend
            verticalAlign="top"
            align="right"
            iconType="circle"
            wrapperStyle={{ paddingBottom: '10px', fontSize: '12px', color: '#94a3b8' }}
          />

          <Area
            type="monotone"
            dataKey="sales_amount"
            name="Revenue"
            stroke="#6366f1"
            strokeWidth={2.5}
            fillOpacity={1}
            fill="url(#colorSales)"
          />

          <Area
            type="monotone"
            dataKey="profit"
            name="Profit"
            stroke="#10b981"
            strokeWidth={2.5}
            fillOpacity={1}
            fill="url(#colorProfit)"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default SalesChart;
