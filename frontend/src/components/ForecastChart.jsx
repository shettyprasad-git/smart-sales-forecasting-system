import React from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceLine,
} from 'recharts';
import { formatNumber, formatDate } from '../utils/formatters';

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-900 border border-slate-700/80 p-3 rounded-xl shadow-xl text-xs space-y-1.5">
        <p className="font-semibold text-slate-300 border-b border-slate-800 pb-1">
          {formatDate(label)}
        </p>
        {payload.map((entry, index) => {
          if (entry.value === null || entry.value === undefined) return null;
          return (
            <div key={`item-${index}`} className="flex items-center justify-between space-x-4">
              <span className="flex items-center space-x-1.5" style={{ color: entry.color }}>
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
                <span className="font-medium text-slate-300">{entry.name}:</span>
              </span>
              <span className="font-bold text-slate-100 font-mono">
                {formatNumber(entry.value)} units
              </span>
            </div>
          );
        })}
      </div>
    );
  }
  return null;
};

const ForecastChart = ({ historical = [], forecast = [], modelName = 'ML Forecast' }) => {
  // Combine historical and forecast into unified chart series
  // Historical data has `quantity`
  // Forecast data has `predicted_quantity`
  
  const recentHistory = historical.slice(-60); // Show last 60 historical days for visual clarity

  const combinedData = [
    ...recentHistory.map((item) => ({
      date: item.date,
      Historical: item.quantity,
      Forecast: null,
    })),
  ];

  if (recentHistory.length > 0 && forecast.length > 0) {
    // Add bridge point from last historical point to first forecast point
    const lastHist = recentHistory[recentHistory.length - 1];
    combinedData.push({
      date: lastHist.date,
      Historical: lastHist.quantity,
      Forecast: lastHist.quantity,
    });
  }

  forecast.forEach((item) => {
    combinedData.push({
      date: item.date,
      Historical: null,
      Forecast: item.predicted_quantity,
    });
  });

  if (combinedData.length === 0) {
    return (
      <div className="h-72 flex items-center justify-center text-slate-500 text-xs italic border border-slate-800/50 rounded-2xl bg-slate-900/30">
        No demand data available for forecast visualization.
      </div>
    );
  }

  const splitDate = recentHistory.length > 0 ? recentHistory[recentHistory.length - 1].date : null;

  return (
    <div className="w-full h-80 pt-2">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={combinedData} margin={{ top: 10, right: 15, left: 10, bottom: 0 }}>
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
            tickFormatter={(val) => formatNumber(val)}
            width={60}
          />

          <Tooltip content={<CustomTooltip />} />

          <Legend
            verticalAlign="top"
            align="right"
            iconType="circle"
            wrapperStyle={{ paddingBottom: '10px', fontSize: '12px', color: '#94a3b8' }}
          />

          {splitDate && (
            <ReferenceLine
              x={splitDate}
              stroke="#6366f1"
              strokeDasharray="4 4"
              label={{
                value: 'Forecast Start',
                fill: '#818cf8',
                fontSize: 10,
                position: 'top',
              }}
            />
          )}

          <Line
            type="monotone"
            dataKey="Historical"
            name="Historical Demand"
            stroke="#38bdf8"
            strokeWidth={2.5}
            dot={false}
            activeDot={{ r: 5 }}
            connectNulls={false}
          />

          <Line
            type="monotone"
            dataKey="Forecast"
            name={`Forecast (${modelName})`}
            stroke="#f43f5e"
            strokeWidth={3}
            strokeDasharray="5 5"
            dot={{ r: 3, fill: '#f43f5e' }}
            activeDot={{ r: 6 }}
            connectNulls={true}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default ForecastChart;
