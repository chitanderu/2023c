#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳健版ARIMA时间序列预测模型
专门处理数据异常值和NaN问题
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class RobustARIMAForecaster:
    def __init__(self):
        self.data = {}
        self.models = {}
        self.forecasts = {}
        
    def load_data(self):
        """加载数据并处理异常值"""
        print("Loading and cleaning data...")
        
        try:
            carbon_df = pd.read_csv('碳排放.csv', encoding='utf-8-sig')
            economic_df = pd.read_csv('经济与能源.csv', encoding='utf-8-sig')
            
            years = list(range(2010, 2021))
            year_cols = [str(year) for year in years]
            
            # GDP数据
            gdp_row = economic_df[economic_df['项目'] == 'GDP']
            if not gdp_row.empty:
                gdp_values = self._extract_numeric_series(gdp_row, year_cols)
                if gdp_values:
                    self.data['GDP'] = pd.Series(gdp_values, index=years)
            
            # 人口数据
            pop_row = economic_df[economic_df['项目'] == '常驻人口']
            if not pop_row.empty:
                pop_values = self._extract_numeric_series(pop_row, year_cols)
                if pop_values:
                    self.data['Population'] = pd.Series(pop_values, index=years)
            
            # 能源消费数据
            energy_row = economic_df[economic_df['项目'] == '能源消费量']
            if not energy_row.empty:
                energy_values = self._extract_numeric_series(energy_row, year_cols)
                if energy_values:
                    self.data['Energy'] = pd.Series(energy_values, index=years)
            
            # 碳排放数据
            carbon_row = carbon_df[carbon_df['项目'] == '碳排放量']
            if not carbon_row.empty:
                carbon_values = self._extract_numeric_series(carbon_row, year_cols)
                if carbon_values:
                    self.data['Carbon'] = pd.Series(carbon_values, index=years)
            
            print(f"Successfully loaded data for variables: {list(self.data.keys())}")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _extract_numeric_series(self, row, year_cols):
        """提取并清理数值序列"""
        values = []
        for col in year_cols:
            try:
                val = row.iloc[0][col]
                if pd.isna(val) or str(val) in ['-', '', 'nan']:
                    values.append(np.nan)
                else:
                    # 清理数值
                    clean_val = str(val).replace(',', '').replace(' ', '')
                    numeric_val = float(clean_val)
                    
                    # 检查异常值
                    if numeric_val < 0 or np.isinf(numeric_val):
                        values.append(np.nan)
                    else:
                        values.append(numeric_val)
            except:
                values.append(np.nan)
        
        # 使用插值填补NaN值
        series = pd.Series(values)
        if series.notna().sum() >= 3:  # 至少3个有效值
            series = series.interpolate(method='linear', limit_direction='both')
            return series.tolist()
        else:
            return None
    
    def build_arima_models(self):
        """构建ARIMA模型"""
        print("\nBuilding ARIMA models...")
        
        for var_name, series in self.data.items():
            print(f"\nProcessing {var_name}...")
            
            # 确保数据质量
            clean_series = series.dropna()
            if len(clean_series) < 5:
                print(f"Insufficient data for {var_name}")
                continue
            
            # 寻找最佳ARIMA参数
            best_model = self._find_best_arima(clean_series, var_name)
            
            if best_model:
                self.models[var_name] = {
                    'model': best_model,
                    'series': clean_series,
                    'aic': best_model.aic
                }
                print(f"{var_name} model built successfully (AIC: {best_model.aic:.2f})")
            else:
                print(f"Failed to build model for {var_name}")
    
    def _find_best_arima(self, series, name, max_order=2):
        """寻找最优ARIMA参数"""
        best_aic = np.inf
        best_model = None
        
        # 简化的网格搜索
        for p in range(max_order + 1):
            for d in range(max_order + 1):
                for q in range(max_order + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic and np.isfinite(fitted_model.aic):
                            best_aic = fitted_model.aic
                            best_model = fitted_model
                    except:
                        continue
        
        return best_model
    
    def generate_forecasts(self, forecast_years):
        """生成预测"""
        print("\nGenerating forecasts...")
        
        for var_name, model_info in self.models.items():
            try:
                model = model_info['model']
                n_steps = len(forecast_years)
                
                # 基础预测
                forecast_result = model.forecast(steps=n_steps)
                
                # 处理可能的异常值
                forecast_result = np.nan_to_num(forecast_result, nan=0.0, 
                                              posinf=0.0, neginf=0.0)
                
                # 确保预测值的合理性
                historical_mean = model_info['series'].mean()
                historical_std = model_info['series'].std()
                
                # 限制预测值在合理范围内
                lower_bound = max(0, historical_mean - 3 * historical_std)
                upper_bound = historical_mean + 5 * historical_std
                
                forecast_result = np.clip(forecast_result, lower_bound, upper_bound)
                
                self.forecasts[var_name] = pd.Series(forecast_result, index=forecast_years)
                print(f"{var_name} forecast completed")
                
            except Exception as e:
                print(f"Failed to forecast {var_name}: {e}")
                
                # 创建简单的线性趋势预测作为备用
                if var_name in self.data:
                    series = self.data[var_name].dropna()
                    if len(series) >= 2:
                        # 计算平均增长率
                        growth_rate = (series.iloc[-1] / series.iloc[0]) ** (1/(len(series)-1)) - 1
                        
                        # 生成线性趋势预测
                        base_value = series.iloc[-1]
                        trend_forecast = []
                        
                        for i in range(len(forecast_years)):
                            predicted_value = base_value * ((1 + growth_rate) ** (i + 1))
                            trend_forecast.append(predicted_value)
                        
                        self.forecasts[var_name] = pd.Series(trend_forecast, index=forecast_years)
                        print(f"Using trend forecast for {var_name}")
    
    def build_regression_models(self, forecast_years):
        """构建回归预测模型"""
        print("\nBuilding regression models...")
        
        # 能源消费回归模型
        if all(var in self.data for var in ['GDP', 'Population', 'Energy']):
            self._build_energy_regression_model(forecast_years)
        
        # 碳排放回归模型
        if all(var in self.data for var in ['GDP', 'Energy', 'Carbon']):
            self._build_carbon_regression_model(forecast_years)
    
    def _build_energy_regression_model(self, forecast_years):
        """构建能源消费回归模型"""
        try:
            # 准备历史数据
            gdp_hist = self.data['GDP'].dropna()
            pop_hist = self.data['Population'].dropna()
            energy_hist = self.data['Energy'].dropna()
            
            # 找到共同时间索引
            common_idx = gdp_hist.index.intersection(pop_hist.index).intersection(energy_hist.index)
            
            if len(common_idx) >= 3:
                X = np.column_stack([gdp_hist.loc[common_idx], pop_hist.loc[common_idx]])
                y = energy_hist.loc[common_idx]
                
                model = LinearRegression()
                model.fit(X, y)
                r2 = model.score(X, y)
                
                print(f"Energy regression model R2: {r2:.3f}")
                
                # 预测
                if 'GDP' in self.forecasts and 'Population' in self.forecasts:
                    gdp_forecast = self.forecasts['GDP']
                    pop_forecast = self.forecasts['Population']
                    
                    X_pred = np.column_stack([gdp_forecast.values, pop_forecast.values])
                    energy_pred = model.predict(X_pred)
                    
                    # 组合预测（如果有ARIMA预测的话）
                    if 'Energy' in self.forecasts:
                        arima_pred = self.forecasts['Energy']
                        combined_pred = (energy_pred + arima_pred.values) / 2
                        self.forecasts['Energy_Combined'] = pd.Series(combined_pred, index=forecast_years)
                    else:
                        self.forecasts['Energy'] = pd.Series(energy_pred, index=forecast_years)
                        
        except Exception as e:
            print(f"Energy regression model failed: {e}")
    
    def _build_carbon_regression_model(self, forecast_years):
        """构建碳排放回归模型"""
        try:
            # 准备历史数据
            gdp_hist = self.data['GDP'].dropna()
            energy_hist = self.data['Energy'].dropna()
            carbon_hist = self.data['Carbon'].dropna()
            
            common_idx = gdp_hist.index.intersection(energy_hist.index).intersection(carbon_hist.index)
            
            if len(common_idx) >= 3:
                X = np.column_stack([gdp_hist.loc[common_idx], energy_hist.loc[common_idx]])
                y = carbon_hist.loc[common_idx]
                
                model = LinearRegression()
                model.fit(X, y)
                r2 = model.score(X, y)
                
                print(f"Carbon regression model R2: {r2:.3f}")
                
                # 预测
                if 'GDP' in self.forecasts:
                    gdp_forecast = self.forecasts['GDP']
                    
                    # 使用最佳能源预测
                    energy_forecast = self.forecasts.get('Energy_Combined', 
                                                        self.forecasts.get('Energy'))
                    
                    if energy_forecast is not None:
                        X_pred = np.column_stack([gdp_forecast.values, energy_forecast.values])
                        carbon_pred = model.predict(X_pred)
                        
                        # 应用政策调整因子
                        policy_adjusted = self._apply_policy_factors(carbon_pred, forecast_years)
                        
                        self.forecasts['Carbon_Policy'] = pd.Series(policy_adjusted, index=forecast_years)
                        
        except Exception as e:
            print(f"Carbon regression model failed: {e}")
    
    def _apply_policy_factors(self, carbon_pred, forecast_years):
        """应用政策调整因子"""
        base_year = 2020
        policy_adjusted = []
        
        for i, year in enumerate(forecast_years):
            years_from_base = year - base_year
            
            # 政策减排强度
            if year <= 2030:
                annual_reduction = 0.005  # 0.5% per year
            elif year <= 2050:
                annual_reduction = 0.02   # 2% per year
            else:
                annual_reduction = 0.035  # 3.5% per year
            
            policy_factor = (1 - annual_reduction) ** years_from_base
            adjusted_value = carbon_pred[i] * policy_factor
            policy_adjusted.append(max(0, adjusted_value))  # 确保非负
        
        return policy_adjusted
    
    def create_sectoral_distribution(self, forecast_years):
        """创建分部门分布预测"""
        print("\nCreating sectoral distribution...")
        
        # 使用最佳能源预测
        energy_forecast = self.forecasts.get('Energy_Combined', 
                                           self.forecasts.get('Energy'))
        
        if energy_forecast is None:
            print("No energy forecast available for sectoral distribution")
            return {}
        
        # 部门占比（2020年基准）
        base_shares = {
            'Industry': 0.75,       # 工业部门
            'Building': 0.12,       # 建筑部门
            'Transport': 0.08,      # 交通部门
            'Residential': 0.04,    # 居民生活
            'Agriculture': 0.01     # 农林部门
        }
        
        # 结构调整趋势（年度变化）
        annual_changes = {
            'Industry': -0.005,     # 工业占比年均下降
            'Building': 0.0025,     # 建筑占比上升
            'Transport': 0.0015,    # 交通占比上升
            'Residential': 0.001,   # 居民占比上升
            'Agriculture': 0.0      # 农林保持稳定
        }
        
        sectoral_forecasts = {}
        base_year = 2020
        
        for i, year in enumerate(forecast_years):
            years_from_base = year - base_year
            total_energy = energy_forecast.iloc[i]
            
            # 计算调整后的占比
            adjusted_shares = {}
            for sector, base_share in base_shares.items():
                change = annual_changes.get(sector, 0)
                adjusted_share = base_share + change * years_from_base
                adjusted_shares[sector] = max(0.005, adjusted_share)
            
            # 标准化占比
            total_share = sum(adjusted_shares.values())
            for sector in adjusted_shares:
                adjusted_shares[sector] /= total_share
                
                # 初始化列表
                if sector not in sectoral_forecasts:
                    sectoral_forecasts[sector] = []
                
                # 分配能源消费
                sectoral_energy = total_energy * adjusted_shares[sector]
                sectoral_forecasts[sector].append(sectoral_energy)
        
        # 转换为Series
        for sector in sectoral_forecasts:
            sectoral_forecasts[sector] = pd.Series(sectoral_forecasts[sector], 
                                                  index=forecast_years)
        
        return sectoral_forecasts
    
    def generate_summary_report(self, sectoral_forecasts):
        """生成汇总报告"""
        print("\nGenerating summary report...")
        
        key_years = [2025, 2030, 2035, 2050, 2060]
        
        report = "ARIMA Time Series Forecasting Results Summary\n"
        report += "=" * 60 + "\n\n"
        
        # 基准年数据
        report += "Base Year (2020) Values:\n"
        report += "-" * 30 + "\n"
        for var_name, series in self.data.items():
            if 2020 in series.index:
                report += f"{var_name}: {series[2020]:.2f}\n"
        
        # 关键年份预测
        report += "\nKey Year Forecasts:\n"
        report += "-" * 30 + "\n"
        
        for year in key_years:
            if any(year in forecast.index for forecast in self.forecasts.values()):
                report += f"\n{year}年:\n"
                
                for var_name, forecast in self.forecasts.items():
                    if year in forecast.index:
                        if 'Policy' in var_name:
                            report += f"  {var_name}: {forecast[year]:.2f} (policy adjusted)\n"
                        else:
                            report += f"  {var_name}: {forecast[year]:.2f}\n"
        
        # 分部门预测（2030年）
        if sectoral_forecasts and 2030 in next(iter(sectoral_forecasts.values())).index:
            report += "\nSectoral Energy Consumption (2030):\n"
            report += "-" * 40 + "\n"
            total_sectoral = 0
            for sector, forecast in sectoral_forecasts.items():
                value = forecast[2030]
                report += f"{sector}: {value:.2f} (10k tce)\n"
                total_sectoral += value
            report += f"Total: {total_sectoral:.2f} (10k tce)\n"
        
        return report
    
    def create_visualizations(self):
        """创建可视化"""
        print("Creating visualizations...")
        
        n_vars = len(self.forecasts)
        if n_vars == 0:
            print("No forecasts available for visualization")
            return
        
        # 动态确定子图布局
        if n_vars <= 2:
            rows, cols = 1, n_vars
        elif n_vars <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 3, 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        if n_vars == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        fig.suptitle('ARIMA Forecasting Results', fontsize=16, fontweight='bold')
        
        for i, (var_name, forecast) in enumerate(self.forecasts.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # 绘制历史数据
            if var_name.replace('_Combined', '').replace('_Policy', '') in self.data:
                historical_name = var_name.replace('_Combined', '').replace('_Policy', '')
                historical = self.data[historical_name]
                ax.plot(historical.index, historical.values, 'o-', 
                       label='Historical', linewidth=2, markersize=6)
            
            # 绘制预测数据
            ax.plot(forecast.index, forecast.values, 's--', 
                   label='Forecast', linewidth=2, alpha=0.8, color='red')
            
            # 添加关键年份标记
            for key_year in [2025, 2030, 2035, 2050]:
                if key_year in forecast.index:
                    ax.axvline(x=key_year, color='gray', linestyle=':', alpha=0.7)
            
            ax.set_title(f'{var_name} Forecast')
            ax.set_xlabel('Year')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(self.forecasts), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('figures/arima_robust_forecast.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, report, sectoral_forecasts):
        """保存结果"""
        print("Saving results...")
        
        # 保存预测数据
        forecast_df = pd.DataFrame(self.forecasts)
        forecast_df.to_csv('outputs/arima_robust_forecasts.csv', encoding='utf-8-sig')
        
        # 保存分部门数据
        if sectoral_forecasts:
            sectoral_df = pd.DataFrame(sectoral_forecasts)
            sectoral_df.to_csv('outputs/sectoral_energy_forecasts.csv', encoding='utf-8-sig')
        
        # 保存报告
        with open('outputs/arima_robust_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("Results saved successfully!")
    
    def run_complete_analysis(self):
        """运行完整分析"""
        print("Starting Robust ARIMA Forecasting Analysis...")
        
        # 1. 加载数据
        if not self.load_data():
            return False
        
        # 2. 构建ARIMA模型
        self.build_arima_models()
        
        # 3. 生成预测
        forecast_years = list(range(2021, 2061))
        self.generate_forecasts(forecast_years)
        
        # 4. 构建回归模型
        self.build_regression_models(forecast_years)
        
        # 5. 分部门分布
        sectoral_forecasts = self.create_sectoral_distribution(forecast_years)
        
        # 6. 生成报告
        report = self.generate_summary_report(sectoral_forecasts)
        
        # 7. 可视化
        self.create_visualizations()
        
        # 8. 保存结果
        self.save_results(report, sectoral_forecasts)
        
        print("\nRobust ARIMA analysis completed successfully!")
        print("Generated files:")
        print("- outputs/arima_robust_forecasts.csv")
        print("- outputs/sectoral_energy_forecasts.csv")
        print("- outputs/arima_robust_report.txt")
        print("- figures/arima_robust_forecast.png")
        
        return True

def main():
    """主函数"""
    import os
    
    # 创建输出目录
    for dirname in ['outputs', 'figures']:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    
    # 运行分析
    forecaster = RobustARIMAForecaster()
    success = forecaster.run_complete_analysis()
    
    return forecaster if success else None

if __name__ == "__main__":
    forecaster = main()