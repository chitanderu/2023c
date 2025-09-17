#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版ARIMA时间序列预测模型
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载数据"""
    carbon_df = pd.read_csv('碳排放.csv', encoding='utf-8-sig')
    economic_df = pd.read_csv('经济与能源.csv', encoding='utf-8-sig')
    
    years = list(range(2010, 2021))
    year_cols = [str(year) for year in years]
    
    data = {}
    
    # GDP
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
    
    # 人口
    pop_row = economic_df[economic_df['项目'] == '常驻人口']
    if not pop_row.empty:
        pop_values = []
        for col in year_cols:
            try:
                val = pop_row.iloc[0][col]
                pop_values.append(float(str(val).replace(',', '')))
            except:
                pop_values.append(np.nan)
        data['population'] = pd.Series(pop_values, index=years)
    
    # 能源消费
    energy_row = economic_df[economic_df['项目'] == '能源消费量']
    if not energy_row.empty:
        energy_values = []
        for col in year_cols:
            try:
                val = energy_row.iloc[0][col]
                energy_values.append(float(str(val).replace(',', '')))
            except:
                energy_values.append(np.nan)
        data['energy'] = pd.Series(energy_values, index=years)
    
    # 碳排放
    carbon_row = carbon_df[carbon_df['项目'] == '碳排放量']
    if not carbon_row.empty:
        carbon_values = []
        for col in year_cols:
            try:
                val = carbon_row.iloc[0][col]
                carbon_values.append(float(str(val).replace(',', '')))
            except:
                carbon_values.append(np.nan)
        data['carbon'] = pd.Series(carbon_values, index=years)
    
    return data

def build_arima_model(series, name, max_p=3, max_d=2, max_q=3):
    """构建ARIMA模型"""
    print(f"Building ARIMA model for {name}...")
    
    best_aic = np.inf
    best_params = None
    best_model = None
    
    # 网格搜索
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted_model = model.fit()
                    aic = fitted_model.aic
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_params = (p, d, q)
                        best_model = fitted_model
                        
                except:
                    continue
    
    print(f"Best params for {name}: ARIMA{best_params}, AIC: {best_aic:.2f}")
    return best_model, best_params

def forecast_with_arima(data, forecast_years):
    """使用ARIMA进行预测"""
    models = {}
    forecasts = {}
    
    variables = ['GDP', 'population', 'energy', 'carbon']
    
    for var in variables:
        if var in data:
            series = data[var].dropna()
            if len(series) >= 5:
                model, params = build_arima_model(series, var)
                if model:
                    # 预测
                    n_steps = len(forecast_years)
                    forecast_result = model.forecast(steps=n_steps)
                    forecast_ci = model.get_forecast(steps=n_steps).conf_int()
                    
                    models[var] = model
                    forecasts[var] = {
                        'forecast': pd.Series(forecast_result, index=forecast_years),
                        'lower_ci': pd.Series(forecast_ci.iloc[:, 0], index=forecast_years),
                        'upper_ci': pd.Series(forecast_ci.iloc[:, 1], index=forecast_years),
                        'historical': series
                    }
    
    return models, forecasts

def build_regression_models(data, forecasts, forecast_years):
    """构建回归模型"""
    print("Building regression models...")
    
    # 能源消费回归模型 (基于GDP和人口)
    if 'GDP' in data and 'population' in data and 'energy' in data:
        gdp_hist = data['GDP'].dropna()
        pop_hist = data['population'].dropna()
        energy_hist = data['energy'].dropna()
        
        common_idx = gdp_hist.index.intersection(pop_hist.index).intersection(energy_hist.index)
        
        if len(common_idx) >= 3:
            X_hist = np.column_stack([gdp_hist.loc[common_idx], pop_hist.loc[common_idx]])
            y_hist = energy_hist.loc[common_idx]
            
            energy_model = LinearRegression()
            energy_model.fit(X_hist, y_hist)
            r2 = energy_model.score(X_hist, y_hist)
            
            print(f"Energy consumption model R2: {r2:.3f}")
            
            # 预测能源消费
            if 'GDP' in forecasts and 'population' in forecasts:
                gdp_forecast = forecasts['GDP']['forecast'].fillna(method='ffill')
                pop_forecast = forecasts['population']['forecast'].fillna(method='ffill')
                
                X_forecast = np.column_stack([
                    gdp_forecast.values,
                    pop_forecast.values
                ])
                energy_pred = energy_model.predict(X_forecast)
                
                # 更新能源消费预测
                if 'energy' not in forecasts:
                    forecasts['energy'] = {}
                forecasts['energy']['regression_forecast'] = pd.Series(energy_pred, index=forecast_years)
    
    # 碳排放回归模型
    if 'GDP' in data and 'energy' in data and 'carbon' in data:
        gdp_hist = data['GDP'].dropna()
        energy_hist = data['energy'].dropna()
        carbon_hist = data['carbon'].dropna()
        
        common_idx = gdp_hist.index.intersection(energy_hist.index).intersection(carbon_hist.index)
        
        if len(common_idx) >= 3:
            X_hist = np.column_stack([gdp_hist.loc[common_idx], energy_hist.loc[common_idx]])
            y_hist = carbon_hist.loc[common_idx]
            
            carbon_model = LinearRegression()
            carbon_model.fit(X_hist, y_hist)
            r2 = carbon_model.score(X_hist, y_hist)
            
            print(f"Carbon emission model R2: {r2:.3f}")
            
            # 预测碳排放
            if 'GDP' in forecasts and 'energy' in forecasts:
                # 使用能源消费的最佳预测
                energy_forecast = forecasts['energy'].get('regression_forecast', 
                                                        forecasts['energy']['forecast'])
                gdp_forecast = forecasts['GDP']['forecast'].fillna(method='ffill')
                energy_forecast = energy_forecast.fillna(method='ffill')
                
                X_forecast = np.column_stack([
                    gdp_forecast.values,
                    energy_forecast.values
                ])
                carbon_pred = carbon_model.predict(X_forecast)
                
                # 应用政策调整
                base_year = 2020
                policy_factors = []
                for year in forecast_years:
                    years_from_base = year - base_year
                    if year <= 2030:
                        annual_reduction = 0.005  # 0.5% per year
                    elif year <= 2050:
                        annual_reduction = 0.02   # 2% per year
                    else:
                        annual_reduction = 0.03   # 3% per year
                    
                    factor = (1 - annual_reduction) ** years_from_base
                    policy_factors.append(factor)
                
                adjusted_carbon_pred = carbon_pred * np.array(policy_factors)
                
                if 'carbon' not in forecasts:
                    forecasts['carbon'] = {}
                forecasts['carbon']['policy_adjusted'] = pd.Series(adjusted_carbon_pred, index=forecast_years)
    
    return forecasts

def create_sectoral_forecasts(forecasts, forecast_years):
    """创建分部门预测"""
    if 'energy' not in forecasts:
        return {}
    
    # 使用最佳能源预测
    total_energy = forecasts['energy'].get('regression_forecast', 
                                          forecasts['energy']['forecast'])
    
    # 2020年基准部门占比
    sector_shares = {
        'industry': 0.75,      # 工业部门
        'building': 0.12,      # 建筑部门  
        'transport': 0.08,     # 交通部门
        'residential': 0.04,   # 居民生活
        'agriculture': 0.01    # 农林部门
    }
    
    # 结构调整趋势 (年度变化率)
    sector_trends = {
        'industry': -0.005,    # 工业占比年均下降0.5%
        'building': 0.002,     # 建筑占比年均上升0.2%
        'transport': 0.002,    # 交通占比年均上升0.2%
        'residential': 0.001,  # 居民占比年均上升0.1%
        'agriculture': 0.0     # 农林保持稳定
    }
    
    sectoral_forecasts = {}
    base_year = 2020
    
    for i, year in enumerate(forecast_years):
        years_from_base = year - base_year
        total_energy_year = total_energy.iloc[i]
        
        # 调整部门占比
        adjusted_shares = {}
        for sector in sector_shares:
            trend = sector_trends[sector]
            adjusted_share = sector_shares[sector] + trend * years_from_base
            adjusted_shares[sector] = max(0.005, adjusted_share)  # 最小0.5%
        
        # 标准化
        total_share = sum(adjusted_shares.values())
        for sector in adjusted_shares:
            adjusted_shares[sector] /= total_share
        
        # 分配能源消费
        for sector in adjusted_shares:
            if sector not in sectoral_forecasts:
                sectoral_forecasts[sector] = []
            sectoral_forecasts[sector].append(total_energy_year * adjusted_shares[sector])
    
    # 转换为Series
    for sector in sectoral_forecasts:
        sectoral_forecasts[sector] = pd.Series(sectoral_forecasts[sector], index=forecast_years)
    
    return sectoral_forecasts

def generate_summary_report(data, forecasts, sectoral_forecasts):
    """生成汇总报告"""
    key_years = [2025, 2030, 2035, 2050, 2060]
    
    report = "ARIMA Time Series Forecasting Results\n"
    report += "="*50 + "\n\n"
    
    report += "Base Year (2020) Values:\n"
    report += "-"*30 + "\n"
    base_year = 2020
    for var, series in data.items():
        if base_year in series.index:
            value = series[base_year]
            report += f"{var}: {value:.2f}\n"
    
    report += "\nKey Year Forecasts:\n"
    report += "-"*30 + "\n"
    
    for year in key_years:
        if year <= max(forecasts[list(forecasts.keys())[0]]['forecast'].index):
            report += f"\n{year}:\n"
            
            for var, forecast_data in forecasts.items():
                # 选择最佳预测
                if 'policy_adjusted' in forecast_data:
                    forecast_val = forecast_data['policy_adjusted'][year]
                elif 'regression_forecast' in forecast_data:
                    forecast_val = forecast_data['regression_forecast'][year]
                else:
                    forecast_val = forecast_data['forecast'][year]
                
                report += f"  {var}: {forecast_val:.2f}\n"
    
    if sectoral_forecasts:
        report += "\nSectoral Energy Consumption (2030):\n"
        report += "-"*40 + "\n"
        for sector, forecast in sectoral_forecasts.items():
            if 2030 in forecast.index:
                report += f"{sector}: {forecast[2030]:.2f}\n"
    
    return report

def create_visualizations(data, forecasts, forecast_years):
    """创建可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('ARIMA Forecast Results', fontsize=16)
    
    variables = ['GDP', 'population', 'energy', 'carbon']
    titles = ['GDP (billion yuan)', 'Population (10k)', 'Energy (10k tce)', 'Carbon (10k tCO2)']
    
    for i, (var, title) in enumerate(zip(variables, titles)):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        if var in forecasts:
            # 历史数据
            historical = forecasts[var]['historical']
            ax.plot(historical.index, historical.values, 'o-', label='Historical', linewidth=2)
            
            # 预测数据
            if 'policy_adjusted' in forecasts[var]:
                forecast_series = forecasts[var]['policy_adjusted']
                label = 'Policy Adjusted Forecast'
            elif 'regression_forecast' in forecasts[var]:
                forecast_series = forecasts[var]['regression_forecast']
                label = 'Regression Forecast'
            else:
                forecast_series = forecasts[var]['forecast']
                label = 'ARIMA Forecast'
            
            ax.plot(forecast_series.index, forecast_series.values, 's--', 
                   label=label, linewidth=2, alpha=0.8)
            
            # 置信区间
            if 'lower_ci' in forecasts[var] and 'upper_ci' in forecasts[var]:
                ax.fill_between(forecast_series.index,
                               forecasts[var]['lower_ci'],
                               forecasts[var]['upper_ci'],
                               alpha=0.3)
            
            # 关键年份标记
            for key_year in [2025, 2030, 2035, 2050]:
                if key_year <= max(forecast_series.index):
                    ax.axvline(x=key_year, color='gray', linestyle=':', alpha=0.7)
            
            ax.set_title(title)
            ax.set_xlabel('Year')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/arima_forecast_simple.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("Starting ARIMA Forecasting...")
    
    # 创建目录
    import os
    for dirname in ['outputs', 'figures']:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    
    # 1. 加载数据
    data = load_data()
    print("Data loaded successfully!")
    
    # 2. 设置预测年份
    forecast_years = list(range(2021, 2061))
    
    # 3. ARIMA预测
    models, forecasts = forecast_with_arima(data, forecast_years)
    
    # 4. 回归模型
    forecasts = build_regression_models(data, forecasts, forecast_years)
    
    # 5. 分部门预测
    sectoral_forecasts = create_sectoral_forecasts(forecasts, forecast_years)
    
    # 6. 生成报告
    report = generate_summary_report(data, forecasts, sectoral_forecasts)
    
    # 7. 保存结果
    with open('outputs/arima_forecast_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 保存预测数据
    forecast_df = pd.DataFrame()
    for var, forecast_data in forecasts.items():
        if 'policy_adjusted' in forecast_data:
            forecast_df[var] = forecast_data['policy_adjusted']
        elif 'regression_forecast' in forecast_data:
            forecast_df[var] = forecast_data['regression_forecast']
        else:
            forecast_df[var] = forecast_data['forecast']
    
    forecast_df.to_csv('outputs/arima_forecast_data.csv', encoding='utf-8-sig')
    
    # 8. 可视化
    create_visualizations(data, forecasts, forecast_years)
    
    print("\nForecasting completed!")
    print("Generated files:")
    print("- outputs/arima_forecast_report.txt")
    print("- outputs/arima_forecast_data.csv") 
    print("- figures/arima_forecast_simple.png")
    
    return data, forecasts, sectoral_forecasts

if __name__ == "__main__":
    data, forecasts, sectoral = main()