#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于ARIMA时间序列模型的区域碳排放量预测系统
包含人口、经济、能源消费量和碳排放量的预测模型
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ARIMAForecastSystem:
    def __init__(self):
        self.models = {}
        self.forecasts = {}
        self.historical_data = {}
        self.sectoral_models = {}
        
    def load_and_prepare_data(self):
        """加载和准备数据"""
        print("正在加载和准备数据...")
        
        try:
            # 读取数据
            carbon_df = pd.read_csv('碳排放.csv', encoding='utf-8-sig')
            economic_df = pd.read_csv('经济与能源.csv', encoding='utf-8-sig')
            
            # 准备年份
            years = list(range(2010, 2021))
            year_cols = [str(year) for year in years]
            
            # 提取核心时间序列数据
            self.historical_data = self._extract_time_series(carbon_df, economic_df, years, year_cols)
            
            print("数据加载完成!")
            return True
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def _extract_time_series(self, carbon_df, economic_df, years, year_cols):
        """提取时间序列数据"""
        data = {}
        
        # 1. GDP数据
        gdp_row = economic_df[economic_df['项目'] == 'GDP']
        if not gdp_row.empty:
            gdp_values = []
            for col in year_cols:
                try:
                    val = gdp_row.iloc[0][col]
                    gdp_values.append(float(str(val).replace(',', '')))
                except:
                    gdp_values.append(np.nan)
            data['GDP'] = pd.Series(gdp_values, index=years)
        
        # 2. 人口数据
        pop_row = economic_df[economic_df['项目'] == '常驻人口']
        if not pop_row.empty:
            pop_values = []
            for col in year_cols:
                try:
                    val = pop_row.iloc[0][col]
                    pop_values.append(float(str(val).replace(',', '')))
                except:
                    pop_values.append(np.nan)
            data['人口'] = pd.Series(pop_values, index=years)
        
        # 3. 能源消费总量
        energy_row = economic_df[economic_df['项目'] == '能源消费量']
        if not energy_row.empty:
            energy_values = []
            for col in year_cols:
                try:
                    val = energy_row.iloc[0][col]
                    energy_values.append(float(str(val).replace(',', '')))
                except:
                    energy_values.append(np.nan)
            data['能源消费总量'] = pd.Series(energy_values, index=years)
        
        # 4. 碳排放总量
        carbon_row = carbon_df[carbon_df['项目'] == '碳排放量']
        if not carbon_row.empty:
            carbon_values = []
            for col in year_cols:
                try:
                    val = carbon_row.iloc[0][col]
                    carbon_values.append(float(str(val).replace(',', '')))
                except:
                    carbon_values.append(np.nan)
            data['碳排放总量'] = pd.Series(carbon_values, index=years)
        
        # 5. 分部门能源消费数据
        sectors = ['农林消费部门', '工业消费部门', '建筑消费部门', '交通消费部门', '居民生活消费']
        
        for sector in sectors:
            # 从经济数据中提取分部门能源消费
            sector_rows = economic_df[
                (economic_df['项目'].str.contains(sector, na=False)) |
                (economic_df['子项'].str.contains(sector, na=False))
            ]
            
            if not sector_rows.empty:
                sector_values = []
                for col in year_cols:
                    total = 0
                    for _, row in sector_rows.iterrows():
                        try:
                            val = row[col]
                            if pd.notna(val) and str(val) != '-':
                                total += float(str(val).replace(',', ''))
                        except:
                            continue
                    sector_values.append(total if total > 0 else np.nan)
                
                if any(pd.notna(sector_values)):
                    data[f'{sector}_能源消费'] = pd.Series(sector_values, index=years)
        
        # 6. 分部门碳排放数据
        for sector in sectors:
            sector_rows = carbon_df[
                (carbon_df['项目'].str.contains(sector, na=False)) |
                (carbon_df['子项'].str.contains(sector, na=False))
            ]
            
            if not sector_rows.empty:
                sector_values = []
                for col in year_cols:
                    total = 0
                    for _, row in sector_rows.iterrows():
                        try:
                            val = row[col]
                            if pd.notna(val) and str(val) != '-':
                                total += float(str(val).replace(',', ''))
                        except:
                            continue
                    sector_values.append(total if total > 0 else np.nan)
                
                if any(pd.notna(sector_values)):
                    data[f'{sector}_碳排放'] = pd.Series(sector_values, index=years)
        
        # 7. 能源结构数据（煤炭、油品、天然气、电力等）
        fuel_types = ['煤炭', '油品', '天然气', '电力']
        
        for fuel in fuel_types:
            # 提取该燃料类型的总消费量
            fuel_rows = economic_df[
                economic_df['细分项'].str.contains(fuel, na=False) if '细分项' in economic_df.columns else pd.Series([False] * len(economic_df))
            ]
            
            if not fuel_rows.empty:
                fuel_values = []
                for col in year_cols:
                    total = 0
                    for _, row in fuel_rows.iterrows():
                        try:
                            val = row[col]
                            if pd.notna(val) and str(val) != '-':
                                total += float(str(val).replace(',', ''))
                        except:
                            continue
                    fuel_values.append(total if total > 0 else np.nan)
                
                if any(pd.notna(fuel_values)):
                    data[f'{fuel}_消费量'] = pd.Series(fuel_values, index=years)
        
        return data
    
    def check_stationarity(self, series, name):
        """检查时间序列的平稳性"""
        # 执行ADF检验
        result = adfuller(series.dropna())
        
        print(f'\n{name} 平稳性检验:')
        print(f'ADF统计量: {result[0]:.6f}')
        print(f'p值: {result[1]:.6f}')
        print(f'临界值:')
        for key, value in result[4].items():
            print(f'\t{key}: {value:.3f}')
        
        # 判断是否平稳
        is_stationary = result[1] <= 0.05
        print(f'结果: {"平稳" if is_stationary else "非平稳"}')
        
        return is_stationary
    
    def find_best_arima_params(self, series, max_p=3, max_d=2, max_q=3):
        """寻找最优ARIMA参数"""
        best_aic = np.inf
        best_params = None
        
        # 首先确定差分阶数
        d = 0
        temp_series = series.copy()
        while d <= max_d:
            if self.check_stationarity(temp_series, f"差分{d}阶后"):
                break
            temp_series = temp_series.diff().dropna()
            d += 1
        
        print(f"\n寻找最优ARIMA参数 (最大差分阶数: {d})...")
        
        # 网格搜索最优参数
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted_model = model.fit()
                    aic = fitted_model.aic
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                        
                except Exception as e:
                    continue
        
        print(f"最优参数: ARIMA{best_params}, AIC: {best_aic:.2f}")
        return best_params
    
    def build_arima_models(self):
        """构建ARIMA模型"""
        print("\n开始构建ARIMA模型...")
        
        # 核心变量
        core_variables = ['GDP', '人口', '能源消费总量', '碳排放总量']
        
        for var_name in core_variables:
            if var_name in self.historical_data:
                series = self.historical_data[var_name].dropna()
                
                if len(series) < 5:  # 数据点太少，无法建模
                    print(f"警告: {var_name} 数据点过少，跳过建模")
                    continue
                
                print(f"\n{'='*50}")
                print(f"正在为 {var_name} 构建ARIMA模型")
                print(f"{'='*50}")
                
                # 检查平稳性
                self.check_stationarity(series, var_name)
                
                # 寻找最优参数
                best_params = self.find_best_arima_params(series)
                
                if best_params:
                    try:
                        # 构建最终模型
                        model = ARIMA(series, order=best_params)
                        fitted_model = model.fit()
                        
                        # 保存模型
                        self.models[var_name] = {
                            'model': fitted_model,
                            'params': best_params,
                            'series': series,
                            'aic': fitted_model.aic,
                            'summary': fitted_model.summary()
                        }
                        
                        print(f"{var_name} 模型构建成功!")
                        
                    except Exception as e:
                        print(f"{var_name} 模型构建失败: {e}")
        
        return self.models
    
    def forecast_core_variables(self, forecast_years):
        """预测核心变量"""
        print("\n开始预测核心变量...")
        
        forecasts = {}
        
        for var_name, model_info in self.models.items():
            print(f"\n预测 {var_name}...")
            
            try:
                # 获取预测步数
                n_steps = len(forecast_years)
                
                # 进行预测
                forecast_result = model_info['model'].forecast(steps=n_steps)
                forecast_ci = model_info['model'].get_forecast(steps=n_steps).conf_int()
                
                # 创建预测序列
                forecast_series = pd.Series(forecast_result, index=forecast_years)
                
                # 保存预测结果
                forecasts[var_name] = {
                    'forecast': forecast_series,
                    'lower_ci': pd.Series(forecast_ci.iloc[:, 0], index=forecast_years),
                    'upper_ci': pd.Series(forecast_ci.iloc[:, 1], index=forecast_years),
                    'historical': model_info['series']
                }
                
                print(f"{var_name} 预测完成!")
                
            except Exception as e:
                print(f"{var_name} 预测失败: {e}")
        
        self.forecasts = forecasts
        return forecasts
    
    def build_energy_consumption_model(self, forecast_years):
        """构建基于人口和GDP的能源消费量预测模型"""
        print("\n构建能源消费量预测模型...")
        
        if 'GDP' not in self.forecasts or '人口' not in self.forecasts:
            print("缺少GDP或人口预测数据，无法构建能源消费模型")
            return None
        
        # 历史数据回归分析
        historical_gdp = self.historical_data['GDP'].dropna()
        historical_pop = self.historical_data['人口'].dropna()
        historical_energy = self.historical_data['能源消费总量'].dropna()
        
        # 找到共同的时间索引
        common_index = historical_gdp.index.intersection(historical_pop.index).intersection(historical_energy.index)
        
        if len(common_index) < 3:
            print("历史数据不足，无法建立回归模型")
            return None
        
        # 准备回归数据
        X_hist = np.column_stack([
            historical_gdp.loc[common_index].values,
            historical_pop.loc[common_index].values
        ])
        y_hist = historical_energy.loc[common_index].values
        
        # 简单线性回归 (可以改为更复杂的模型)
        from sklearn.linear_model import LinearRegression
        energy_model = LinearRegression()
        energy_model.fit(X_hist, y_hist)
        
        # 计算历史拟合度
        y_pred_hist = energy_model.predict(X_hist)
        r2_score = energy_model.score(X_hist, y_hist)
        
        print(f"能源消费模型 R2: {r2_score:.3f}")
        
        # 使用预测的GDP和人口数据预测能源消费
        gdp_forecast = self.forecasts['GDP']['forecast']
        pop_forecast = self.forecasts['人口']['forecast']
        
        X_forecast = np.column_stack([
            gdp_forecast.values,
            pop_forecast.values
        ])
        
        energy_forecast = energy_model.predict(X_forecast)
        energy_forecast_series = pd.Series(energy_forecast, index=forecast_years)
        
        # 更新能源消费预测结果
        if '能源消费总量' in self.forecasts:
            # 如果已有ARIMA预测，可以进行组合
            arima_forecast = self.forecasts['能源消费总量']['forecast']
            # 简单平均组合
            combined_forecast = (energy_forecast_series + arima_forecast) / 2
            
            self.forecasts['能源消费总量']['regression_forecast'] = energy_forecast_series
            self.forecasts['能源消费总量']['combined_forecast'] = combined_forecast
        else:
            self.forecasts['能源消费总量'] = {
                'forecast': energy_forecast_series,
                'regression_forecast': energy_forecast_series,
                'model': energy_model,
                'r2': r2_score
            }
        
        return energy_model
    
    def build_sectoral_energy_models(self, forecast_years):
        """构建分部门能源消费模型"""
        print("\n构建分部门能源消费模型...")
        
        if '能源消费总量' not in self.forecasts:
            print("缺少总能源消费预测，无法分配到各部门")
            return None
        
        total_energy_forecast = self.forecasts['能源消费总量']['forecast']
        sectors = ['农林消费部门', '工业消费部门', '建筑消费部门', '交通消费部门', '居民生活消费']
        
        sectoral_forecasts = {}
        
        # 计算历史各部门占比
        base_year = 2020
        sectoral_shares = {}
        total_base = 0
        
        for sector in sectors:
            sector_key = f'{sector}_能源消费'
            if sector_key in self.historical_data:
                sector_data = self.historical_data[sector_key].dropna()
                if base_year in sector_data.index:
                    sectoral_shares[sector] = sector_data[base_year]
                    total_base += sector_data[base_year]
        
        # 标准化占比
        if total_base > 0:
            for sector in sectoral_shares:
                sectoral_shares[sector] = sectoral_shares[sector] / total_base
        
        # 考虑结构调整趋势
        structural_trends = {
            '工业消费部门': -0.005,  # 年均下降0.5个百分点
            '建筑消费部门': 0.002,   # 年均增长0.2个百分点
            '交通消费部门': 0.002,   # 年均增长0.2个百分点
            '居民生活消费': 0.001,   # 年均增长0.1个百分点
            '农林消费部门': 0.0      # 保持稳定
        }
        
        # 预测各部门能源消费
        for i, year in enumerate(forecast_years):
            years_from_base = year - base_year
            
            # 调整后的部门占比
            adjusted_shares = {}
            for sector in sectoral_shares:
                if sector in structural_trends:
                    trend = structural_trends[sector]
                    adjusted_share = sectoral_shares[sector] + trend * years_from_base
                    adjusted_shares[sector] = max(0.01, adjusted_share)  # 最小1%
                else:
                    adjusted_shares[sector] = sectoral_shares[sector]
            
            # 标准化调整后的占比
            total_adjusted = sum(adjusted_shares.values())
            if total_adjusted > 0:
                for sector in adjusted_shares:
                    adjusted_shares[sector] = adjusted_shares[sector] / total_adjusted
            
            # 分配总能源消费到各部门
            total_energy = total_energy_forecast.iloc[i]
            for sector in adjusted_shares:
                if sector not in sectoral_forecasts:
                    sectoral_forecasts[sector] = []
                sectoral_forecasts[sector].append(total_energy * adjusted_shares[sector])
        
        # 转换为Series格式
        for sector in sectoral_forecasts:
            sectoral_forecasts[sector] = pd.Series(sectoral_forecasts[sector], index=forecast_years)
        
        self.sectoral_forecasts = sectoral_forecasts
        return sectoral_forecasts
    
    def build_carbon_emission_model(self, forecast_years):
        """构建碳排放预测模型"""
        print("\n构建碳排放预测模型...")
        
        # 方法1: 基于总量的回归模型
        if all(var in self.forecasts for var in ['GDP', '人口', '能源消费总量']):
            carbon_forecasts = self._build_aggregate_carbon_model(forecast_years)
        
        # 方法2: 基于分部门的模型
        if hasattr(self, 'sectoral_forecasts'):
            sectoral_carbon_forecasts = self._build_sectoral_carbon_model(forecast_years)
        
        return carbon_forecasts
    
    def _build_aggregate_carbon_model(self, forecast_years):
        """总量碳排放模型"""
        # 历史数据回归
        historical_gdp = self.historical_data['GDP'].dropna()
        historical_energy = self.historical_data['能源消费总量'].dropna()
        historical_carbon = self.historical_data['碳排放总量'].dropna()
        
        common_index = historical_gdp.index.intersection(historical_energy.index).intersection(historical_carbon.index)
        
        if len(common_index) < 3:
            print("历史数据不足，使用ARIMA预测结果")
            return self.forecasts.get('碳排放总量', None)
        
        # 准备回归数据
        X_hist = np.column_stack([
            historical_gdp.loc[common_index].values,
            historical_energy.loc[common_index].values
        ])
        y_hist = historical_carbon.loc[common_index].values
        
        # 构建回归模型
        from sklearn.linear_model import LinearRegression
        carbon_model = LinearRegression()
        carbon_model.fit(X_hist, y_hist)
        
        r2_score = carbon_model.score(X_hist, y_hist)
        print(f"碳排放总量模型 R2: {r2_score:.3f}")
        
        # 预测
        gdp_forecast = self.forecasts['GDP']['forecast']
        energy_forecast = self.forecasts['能源消费总量']['forecast']
        
        X_forecast = np.column_stack([
            gdp_forecast.values,
            energy_forecast.values
        ])
        
        carbon_forecast = carbon_model.predict(X_forecast)
        
        # 应用政策调整因子
        policy_factors = self._calculate_policy_adjustment_factors(forecast_years)
        adjusted_carbon_forecast = carbon_forecast * policy_factors
        
        carbon_forecast_series = pd.Series(adjusted_carbon_forecast, index=forecast_years)
        
        # 更新预测结果
        if '碳排放总量' not in self.forecasts:
            self.forecasts['碳排放总量'] = {}
        
        self.forecasts['碳排放总量']['regression_forecast'] = carbon_forecast_series
        self.forecasts['碳排放总量']['policy_adjusted_forecast'] = carbon_forecast_series
        
        return carbon_forecast_series
    
    def _build_sectoral_carbon_model(self, forecast_years):
        """分部门碳排放模型"""
        if not hasattr(self, 'sectoral_forecasts'):
            return None
        
        sectoral_carbon_forecasts = {}
        
        # 各部门碳排放系数(基准年2020)
        base_emission_factors = {
            '工业消费部门': 2.8,     # tCO2/tce
            '建筑消费部门': 2.2,     # tCO2/tce  
            '交通消费部门': 2.4,     # tCO2/tce
            '居民生活消费': 2.0,     # tCO2/tce
            '农林消费部门': 1.8      # tCO2/tce
        }
        
        # 排放因子改善趋势 (年均下降率)
        emission_factor_trends = {
            '工业消费部门': -0.02,   # 年均下降2%
            '建筑消费部门': -0.015,  # 年均下降1.5%
            '交通消费部门': -0.025,  # 年均下降2.5%
            '居民生活消费': -0.01,   # 年均下降1%
            '农林消费部门': -0.005   # 年均下降0.5%
        }
        
        base_year = 2020
        
        for sector, energy_forecast in self.sectoral_forecasts.items():
            if sector in base_emission_factors:
                carbon_emissions = []
                
                for i, year in enumerate(forecast_years):
                    years_from_base = year - base_year
                    
                    # 调整后的排放因子
                    base_factor = base_emission_factors[sector]
                    trend = emission_factor_trends[sector]
                    adjusted_factor = base_factor * ((1 + trend) ** years_from_base)
                    
                    # 碳排放 = 能源消费 * 排放因子
                    carbon_emission = energy_forecast.iloc[i] * adjusted_factor
                    carbon_emissions.append(carbon_emission)
                
                sectoral_carbon_forecasts[sector] = pd.Series(carbon_emissions, index=forecast_years)
        
        # 计算总碳排放
        if sectoral_carbon_forecasts:
            total_sectoral_carbon = pd.DataFrame(sectoral_carbon_forecasts).sum(axis=1)
            
            if '碳排放总量' not in self.forecasts:
                self.forecasts['碳排放总量'] = {}
            
            self.forecasts['碳排放总量']['sectoral_forecast'] = total_sectoral_carbon
        
        self.sectoral_carbon_forecasts = sectoral_carbon_forecasts
        return sectoral_carbon_forecasts
    
    def _calculate_policy_adjustment_factors(self, forecast_years):
        """计算政策调整因子"""
        base_year = 2020
        factors = []
        
        for year in forecast_years:
            years_from_base = year - base_year
            
            # 基于中国双碳目标的政策强度
            if year <= 2030:  # 碳达峰期
                # 逐步加强的减排政策
                annual_reduction = 0.01  # 年均1%的政策减排效应
            elif year <= 2050:  # 快速减排期
                # 强化减排政策
                annual_reduction = 0.025  # 年均2.5%的政策减排效应  
            else:  # 碳中和期
                # 深度减排政策
                annual_reduction = 0.04   # 年均4%的政策减排效应
            
            # 累积政策效应
            cumulative_factor = (1 - annual_reduction) ** years_from_base
            factors.append(cumulative_factor)
        
        return np.array(factors)
    
    def generate_forecasts_summary(self, forecast_years):
        """生成预测结果汇总"""
        print("\n生成预测结果汇总...")
        
        # 关键时间节点
        key_years = [2025, 2030, 2035, 2050, 2060]
        available_years = [year for year in key_years if year in forecast_years]
        
        summary = {
            'base_year_2020': {},
            'forecasts': {}
        }
        
        # 基准年数据
        base_year = 2020
        for var_name in ['GDP', '人口', '能源消费总量', '碳排放总量']:
            if var_name in self.historical_data:
                if base_year in self.historical_data[var_name].index:
                    summary['base_year_2020'][var_name] = self.historical_data[var_name][base_year]
        
        # 预测数据
        for year in available_years:
            summary['forecasts'][year] = {}
            
            for var_name, forecast_data in self.forecasts.items():
                if 'policy_adjusted_forecast' in forecast_data:
                    forecast_series = forecast_data['policy_adjusted_forecast']
                elif 'combined_forecast' in forecast_data:
                    forecast_series = forecast_data['combined_forecast']
                else:
                    forecast_series = forecast_data.get('forecast')
                
                if forecast_series is not None and year in forecast_series.index:
                    summary['forecasts'][year][var_name] = forecast_series[year]
        
        # 计算增长率
        for year in available_years:
            if year in summary['forecasts']:
                summary['forecasts'][year]['增长率'] = {}
                for var_name in ['GDP', '人口', '能源消费总量', '碳排放总量']:
                    if var_name in summary['base_year_2020'] and var_name in summary['forecasts'][year]:
                        base_value = summary['base_year_2020'][var_name]
                        forecast_value = summary['forecasts'][year][var_name]
                        if base_value > 0:
                            growth_rate = ((forecast_value / base_value) ** (1/(year-2020)) - 1) * 100
                            summary['forecasts'][year]['增长率'][var_name] = growth_rate
        
        return summary
    
    def create_forecast_visualizations(self, forecast_years):
        """创建预测可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ARIMA时间序列预测结果', fontsize=16, fontweight='bold')
        
        variables = ['GDP', '人口', '能源消费总量', '碳排放总量']
        units = ['亿元', '万人', '万tce', '万tCO2']
        
        for i, (var_name, unit) in enumerate(zip(variables, units)):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if var_name in self.forecasts:
                forecast_data = self.forecasts[var_name]
                historical = forecast_data.get('historical')
                
                # 选择最佳预测结果
                if 'policy_adjusted_forecast' in forecast_data:
                    forecast = forecast_data['policy_adjusted_forecast']
                elif 'combined_forecast' in forecast_data:
                    forecast = forecast_data['combined_forecast']
                else:
                    forecast = forecast_data.get('forecast')
                
                # 绘制历史数据
                if historical is not None:
                    ax.plot(historical.index, historical.values, 'o-', 
                           label='历史数据', linewidth=2, markersize=6, color='blue')
                
                # 绘制预测数据
                if forecast is not None:
                    ax.plot(forecast.index, forecast.values, 's--', 
                           label='预测数据', linewidth=2, markersize=4, color='red')
                    
                    # 绘制置信区间（如果有）
                    if 'lower_ci' in forecast_data and 'upper_ci' in forecast_data:
                        ax.fill_between(forecast.index, 
                                      forecast_data['lower_ci'].values,
                                      forecast_data['upper_ci'].values,
                                      alpha=0.3, color='red')
                
                # 添加关键年份标记
                key_years = [2025, 2030, 2035, 2050]
                for key_year in key_years:
                    if key_year in forecast.index:
                        ax.axvline(x=key_year, color='gray', linestyle=':', alpha=0.7)
                        ax.text(key_year, ax.get_ylim()[1]*0.9, str(key_year), 
                               rotation=90, ha='right', va='top', fontsize=8)
                
                ax.set_title(f'{var_name}预测', fontsize=12, fontweight='bold')
                ax.set_xlabel('年份')
                ax.set_ylabel(f'{var_name}({unit})')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figures/arima_forecast_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_forecast_results(self, summary):
        """保存预测结果"""
        print("保存预测结果...")
        
        # 保存详细预测数据
        forecast_df = pd.DataFrame()
        
        for var_name, forecast_data in self.forecasts.items():
            # 选择最佳预测结果
            if 'policy_adjusted_forecast' in forecast_data:
                forecast_series = forecast_data['policy_adjusted_forecast']
            elif 'combined_forecast' in forecast_data:
                forecast_series = forecast_data['combined_forecast']
            else:
                forecast_series = forecast_data.get('forecast')
            
            if forecast_series is not None:
                forecast_df[var_name] = forecast_series
        
        forecast_df.to_csv('outputs/arima_forecast_results.csv', encoding='utf-8-sig')
        
        # 保存汇总报告
        with open('outputs/arima_forecast_summary.txt', 'w', encoding='utf-8') as f:
            f.write("ARIMA时间序列预测模型结果汇总\n")
            f.write("="*50 + "\n\n")
            
            f.write("基准年(2020年)数据:\n")
            f.write("-"*30 + "\n")
            for var, value in summary['base_year_2020'].items():
                f.write(f"{var}: {value:.2f}\n")
            
            f.write("\n关键年份预测结果:\n")
            f.write("-"*30 + "\n")
            
            for year, data in summary['forecasts'].items():
                f.write(f"\n{year}年:\n")
                for var, value in data.items():
                    if var != '增长率':
                        f.write(f"  {var}: {value:.2f}\n")
                
                if '增长率' in data:
                    f.write("  年均增长率:\n")
                    for var, rate in data['增长率'].items():
                        f.write(f"    {var}: {rate:.2f}%\n")
        
        print("预测结果已保存!")
    
    def run_complete_forecast(self):
        """运行完整的预测流程"""
        print("开始运行完整的ARIMA预测流程...")
        
        # 1. 加载数据
        if not self.load_and_prepare_data():
            return False
        
        # 2. 构建ARIMA模型
        self.build_arima_models()
        
        # 3. 设置预测年份
        forecast_years = list(range(2021, 2061))
        
        # 4. 预测核心变量
        self.forecast_core_variables(forecast_years)
        
        # 5. 构建能源消费预测模型
        self.build_energy_consumption_model(forecast_years)
        
        # 6. 构建分部门能源消费模型
        self.build_sectoral_energy_models(forecast_years)
        
        # 7. 构建碳排放预测模型
        self.build_carbon_emission_model(forecast_years)
        
        # 8. 生成结果汇总
        summary = self.generate_forecasts_summary(forecast_years)
        
        # 9. 创建可视化
        self.create_forecast_visualizations(forecast_years)
        
        # 10. 保存结果
        self.save_forecast_results(summary)
        
        print("\nARIMA预测流程完成!")
        return True

def main():
    """主函数"""
    import os
    
    # 创建输出目录
    for dirname in ['outputs', 'figures']:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    
    # 初始化预测系统
    forecast_system = ARIMAForecastSystem()
    
    # 运行完整预测
    success = forecast_system.run_complete_forecast()
    
    if success:
        print("\n分析完成！生成的文件：")
        print("- outputs/arima_forecast_results.csv (详细预测数据)")
        print("- outputs/arima_forecast_summary.txt (预测结果汇总)")
        print("- figures/arima_forecast_results.png (预测可视化)")
    else:
        print("预测流程失败，请检查数据和参数设置。")
    
    return forecast_system

if __name__ == "__main__":
    system = main()