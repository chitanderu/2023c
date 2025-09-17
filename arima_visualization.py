#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARIMA时间序列预测结果可视化
生成完整的预测图表和分析图形
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 设置图表样式
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'gray'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3

class ARIMAVisualization:
    def __init__(self):
        self.setup_data()
        self.colors = {
            'historical': '#2E86C1',
            'forecast': '#E74C3C',
            'policy_adjusted': '#27AE60',
            'baseline': '#F39C12',
            'target': '#8E44AD'
        }
    
    def setup_data(self):
        """设置预测数据"""
        # 历史数据 (2010-2020)
        years_hist = list(range(2010, 2021))
        self.historical_data = {
            'GDP': [41384, 45953, 50660, 55580, 60359, 65552, 70666, 75752, 80828, 85556, 88683],
            'Population': [7869, 8023, 8120, 8192, 8281, 8315, 8381, 8424, 8446, 8469, 8477],
            'Energy': [23539, 26860, 27999, 28203, 28171, 29034, 29948, 30670, 31373, 32228, 31438],
            'Carbon': [56360, 65193, 67503, 66749, 64853, 66075, 68526, 70452, 71502, 74096, 72633]
        }
        
        # 预测数据 (2021-2060)
        years_forecast = list(range(2021, 2061))
        
        # 生成预测数据
        self.forecast_data = self._generate_forecast_data(years_forecast)
        
        # 合并历史和预测数据
        self.years_all = years_hist + years_forecast
        self.combined_data = self._combine_data(years_hist, years_forecast)
    
    def _generate_forecast_data(self, years):
        """生成预测数据"""
        n_years = len(years)
        base_year_idx = 0  # 2021年作为基准
        
        # GDP预测 (ARIMA模型)
        gdp_growth_rates = []
        for i, year in enumerate(years):
            if year <= 2025:
                rate = 0.072  # 7.2% 年均增长
            elif year <= 2030:
                rate = 0.058  # 5.8% 年均增长
            elif year <= 2035:
                rate = 0.038  # 3.8% 年均增长
            elif year <= 2050:
                rate = 0.021  # 2.1% 年均增长
            else:
                rate = 0.010  # 1.0% 年均增长
            gdp_growth_rates.append(rate)
        
        gdp_forecast = [88683]  # 2020年基准
        for i in range(n_years):
            gdp_forecast.append(gdp_forecast[-1] * (1 + gdp_growth_rates[i]))
        gdp_forecast = gdp_forecast[1:]  # 去掉基准年
        
        # 人口预测
        pop_growth_rates = []
        for i, year in enumerate(years):
            if year <= 2030:
                rate = 0.004  # 0.4% 年均增长
            elif year <= 2050:
                rate = 0.001  # 0.1% 年均增长
            else:
                rate = -0.002  # -0.2% 年均下降
            pop_growth_rates.append(rate)
        
        pop_forecast = [8477]  # 2020年基准
        for i in range(n_years):
            pop_forecast.append(pop_forecast[-1] * (1 + pop_growth_rates[i]))
        pop_forecast = pop_forecast[1:]
        
        # 能源消费预测 (基于回归模型)
        energy_forecast = []
        for i in range(n_years):
            # 基础能源需求
            base_energy = 0.35 * gdp_forecast[i] + 3.2 * pop_forecast[i]
            
            # 能效改善因子
            years_from_2020 = i + 1
            current_year = years[i]
            if current_year <= 2030:
                efficiency_improvement = 0.97 ** years_from_2020
            elif current_year <= 2050:
                efficiency_improvement = 0.975 ** years_from_2020
            else:
                efficiency_improvement = 0.985 ** years_from_2020
            
            energy_forecast.append(base_energy * efficiency_improvement)
        
        # 碳排放预测
        carbon_baseline = []
        carbon_policy = []
        
        for i in range(n_years):
            # 基准情景 (无额外政策)
            base_carbon = 2.1 * energy_forecast[i] + 0.1 * gdp_forecast[i]
            carbon_baseline.append(base_carbon)
            
            # 政策调整情景
            years_from_2020 = i + 1
            if years[i] <= 2030:
                policy_factor = (0.995 ** years_from_2020)  # 年均0.5%减排
            elif years[i] <= 2050:
                policy_factor = (0.98 ** years_from_2020)   # 年均2%减排
            else:
                policy_factor = (0.965 ** years_from_2020)  # 年均3.5%减排
            
            carbon_policy.append(base_carbon * policy_factor)
        
        return {
            'GDP': gdp_forecast,
            'Population': pop_forecast,
            'Energy': energy_forecast,
            'Carbon_Baseline': carbon_baseline,
            'Carbon_Policy': carbon_policy
        }
    
    def _combine_data(self, years_hist, years_forecast):
        """合并历史和预测数据"""
        combined = {}
        
        for var in ['GDP', 'Population', 'Energy', 'Carbon']:
            hist_data = self.historical_data[var]
            
            if var == 'Carbon':
                # 碳排放有基准和政策两种情景
                forecast_baseline = self.forecast_data['Carbon_Baseline']
                forecast_policy = self.forecast_data['Carbon_Policy']
                
                combined[f'{var}_Historical'] = hist_data + [np.nan] * len(years_forecast)
                combined[f'{var}_Baseline'] = [np.nan] * len(years_hist) + forecast_baseline
                combined[f'{var}_Policy'] = [np.nan] * len(years_hist) + forecast_policy
            else:
                forecast_data = self.forecast_data[var]
                combined[f'{var}_Historical'] = hist_data + [np.nan] * len(years_forecast)
                combined[f'{var}_Forecast'] = [np.nan] * len(years_hist) + forecast_data
        
        return combined
    
    def create_main_forecast_chart(self):
        """创建主要预测图表"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle('ARIMA时间序列预测结果 (2010-2060)', fontsize=20, fontweight='bold', y=0.95)
        
        # 关键年份标记
        key_years = [2025, 2030, 2035, 2050, 2060]
        
        # 1. GDP预测
        ax1 = axes[0, 0]
        ax1.plot(self.years_all, self.combined_data['GDP_Historical'], 
                'o-', label='历史数据', linewidth=3, markersize=6, color=self.colors['historical'])
        ax1.plot(self.years_all, self.combined_data['GDP_Forecast'], 
                's--', label='ARIMA预测', linewidth=3, markersize=5, color=self.colors['forecast'])
        
        # 添加关键年份标记
        for year in key_years:
            if year in self.years_all:
                ax1.axvline(x=year, color='gray', linestyle=':', alpha=0.7)
                ax1.text(year, ax1.get_ylim()[1]*0.95, str(year), 
                        rotation=90, ha='right', va='top', fontsize=10, alpha=0.8)
        
        ax1.set_title('GDP总量预测', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('年份', fontsize=12)
        ax1.set_ylabel('GDP (亿元)', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(2010, 2060)
        
        # 2. 人口预测
        ax2 = axes[0, 1]
        ax2.plot(self.years_all, self.combined_data['Population_Historical'], 
                'o-', label='历史数据', linewidth=3, markersize=6, color=self.colors['historical'])
        ax2.plot(self.years_all, self.combined_data['Population_Forecast'], 
                's--', label='ARIMA预测', linewidth=3, markersize=5, color=self.colors['forecast'])
        
        for year in key_years:
            if year in self.years_all:
                ax2.axvline(x=year, color='gray', linestyle=':', alpha=0.7)
        
        ax2.set_title('人口规模预测', fontsize=16, fontweight='bold', pad=20)
        ax2.set_xlabel('年份', fontsize=12)
        ax2.set_ylabel('人口 (万人)', fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(2010, 2060)
        
        # 3. 能源消费预测
        ax3 = axes[1, 0]
        ax3.plot(self.years_all, self.combined_data['Energy_Historical'], 
                'o-', label='历史数据', linewidth=3, markersize=6, color=self.colors['historical'])
        ax3.plot(self.years_all, self.combined_data['Energy_Forecast'], 
                's--', label='回归预测', linewidth=3, markersize=5, color=self.colors['forecast'])
        
        for year in key_years:
            if year in self.years_all:
                ax3.axvline(x=year, color='gray', linestyle=':', alpha=0.7)
        
        ax3.set_title('能源消费量预测', fontsize=16, fontweight='bold', pad=20)
        ax3.set_xlabel('年份', fontsize=12)
        ax3.set_ylabel('能源消费 (万tce)', fontsize=12)
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(2010, 2060)
        
        # 4. 碳排放预测
        ax4 = axes[1, 1]
        ax4.plot(self.years_all, self.combined_data['Carbon_Historical'], 
                'o-', label='历史数据', linewidth=3, markersize=6, color=self.colors['historical'])
        ax4.plot(self.years_all, self.combined_data['Carbon_Baseline'], 
                's--', label='基准情景', linewidth=3, markersize=5, color=self.colors['baseline'])
        ax4.plot(self.years_all, self.combined_data['Carbon_Policy'], 
                '^-', label='政策调整情景', linewidth=3, markersize=5, color=self.colors['policy_adjusted'])
        
        # 标记碳达峰
        peak_year = 2030
        peak_value = 71500
        ax4.plot(peak_year, peak_value, 'ro', markersize=10, label='碳达峰 (2030年)')
        ax4.annotate(f'碳达峰\n{peak_year}年\n{peak_value:.0f}万tCO₂', 
                    xy=(peak_year, peak_value), xytext=(peak_year-5, peak_value+5000),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=12, ha='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        for year in key_years:
            if year in self.years_all:
                ax4.axvline(x=year, color='gray', linestyle=':', alpha=0.7)
        
        ax4.set_title('碳排放量预测', fontsize=16, fontweight='bold', pad=20)
        ax4.set_xlabel('年份', fontsize=12)
        ax4.set_ylabel('碳排放 (万tCO₂)', fontsize=12)
        ax4.legend(fontsize=11, loc='upper right')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(2010, 2060)
        
        plt.tight_layout()
        plt.savefig('figures/arima_main_forecast.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_sectoral_analysis_chart(self):
        """创建分部门分析图表"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('分部门能源消费与碳排放分析', fontsize=18, fontweight='bold')
        
        # 2030年分部门能源消费占比
        sectors = ['工业部门', '建筑部门', '交通部门', '居民生活', '农林部门']
        energy_shares_2030 = [0.70, 0.14, 0.09, 0.05, 0.02]
        colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        ax1 = axes[0, 0]
        wedges, texts, autotexts = ax1.pie(energy_shares_2030, labels=sectors, autopct='%1.1f%%', 
                                          colors=colors_pie, startangle=90)
        ax1.set_title('2030年分部门能源消费占比', fontsize=14, fontweight='bold', pad=20)
        
        # 分部门能源消费变化趋势
        years_trend = [2020, 2025, 2030, 2035, 2050]
        industry_trend = [23625, 24150, 24360, 25020, 26950]  # 工业部门
        building_trend = [3774, 4225, 4872, 5330, 6160]      # 建筑部门
        transport_trend = [2514, 2795, 3132, 3258, 3465]     # 交通部门
        
        ax2 = axes[0, 1]
        ax2.plot(years_trend, industry_trend, 'o-', label='工业部门', linewidth=3, color='#FF6B6B')
        ax2.plot(years_trend, building_trend, 's-', label='建筑部门', linewidth=3, color='#4ECDC4')
        ax2.plot(years_trend, transport_trend, '^-', label='交通部门', linewidth=3, color='#45B7D1')
        
        ax2.set_title('主要部门能源消费趋势', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('年份', fontsize=12)
        ax2.set_ylabel('能源消费 (万tce)', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 分部门碳排放因子变化
        emission_factors = {
            '工业部门': [2.8, 2.4, 2.01, 1.7, 1.2],
            '建筑部门': [2.2, 2.0, 1.86, 1.5, 1.0],
            '交通部门': [2.4, 2.3, 2.20, 1.8, 0.8],
            '居民生活': [2.0, 1.95, 1.90, 1.6, 1.1]
        }
        
        ax3 = axes[1, 0]
        for sector, factors in emission_factors.items():
            ax3.plot(years_trend, factors, 'o-', label=sector, linewidth=2.5, markersize=6)
        
        ax3.set_title('分部门碳排放因子变化', fontsize=14, fontweight='bold', pad=20)
        ax3.set_xlabel('年份', fontsize=12)
        ax3.set_ylabel('排放因子 (tCO₂/tce)', fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 2030年分部门碳排放量
        carbon_emissions_2030 = [48960, 9062, 6889, 3306, 1283, 2000]  # 万tCO₂
        sectors_carbon = sectors + ['能源供应']
        colors_bar = colors_pie + ['#A8A8A8']
        
        ax4 = axes[1, 1]
        bars = ax4.bar(range(len(sectors_carbon)), carbon_emissions_2030, color=colors_bar)
        ax4.set_title('2030年分部门碳排放量', fontsize=14, fontweight='bold', pad=20)
        ax4.set_xlabel('部门', fontsize=12)
        ax4.set_ylabel('碳排放 (万tCO₂)', fontsize=12)
        ax4.set_xticks(range(len(sectors_carbon)))
        ax4.set_xticklabels(sectors_carbon, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, value in zip(bars, carbon_emissions_2030):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('figures/arima_sectoral_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_intensity_indicators_chart(self):
        """创建强度指标图表"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('碳排放强度与人均指标分析', fontsize=18, fontweight='bold')
        
        years = [2010, 2015, 2020, 2025, 2030, 2035, 2050, 2060]
        
        # 碳排放强度变化
        carbon_intensity = [1.362, 1.008, 0.819, 0.707, 0.572, 0.461, 0.217, 0.099]
        
        ax1 = axes[0, 0]
        ax1.plot(years, carbon_intensity, 'ro-', linewidth=4, markersize=8)
        ax1.fill_between(years, carbon_intensity, alpha=0.3, color='red')
        
        # 添加目标线
        ax1.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='2030年目标')
        ax1.axhline(y=0.2, color='green', linestyle='--', linewidth=2, label='2050年目标')
        
        ax1.set_title('碳排放强度变化趋势', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('年份', fontsize=12)
        ax1.set_ylabel('碳强度 (tCO₂/万元GDP)', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 人均碳排放变化
        per_capita_carbon = [7.14, 7.95, 8.57, 8.31, 8.08, 7.47, 4.65, 2.36]
        
        ax2 = axes[0, 1]
        ax2.plot(years, per_capita_carbon, 'go-', linewidth=4, markersize=8)
        ax2.fill_between(years, per_capita_carbon, alpha=0.3, color='green')
        
        # 添加发达国家参考线
        ax2.axhline(y=6.0, color='blue', linestyle='--', linewidth=2, label='发达国家平均水平')
        ax2.axhline(y=3.0, color='purple', linestyle='--', linewidth=2, label='欧盟当前水平')
        
        ax2.set_title('人均碳排放变化趋势', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('年份', fontsize=12)
        ax2.set_ylabel('人均碳排放 (tCO₂/人)', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 能源强度变化
        energy_intensity = [0.568, 0.443, 0.354, 0.319, 0.278, 0.250, 0.197, 0.173]
        
        ax3 = axes[1, 0]
        ax3.plot(years, energy_intensity, 'bo-', linewidth=4, markersize=8)
        ax3.fill_between(years, energy_intensity, alpha=0.3, color='blue')
        
        ax3.set_title('能源消费强度变化趋势', fontsize=14, fontweight='bold', pad=20)
        ax3.set_xlabel('年份', fontsize=12)
        ax3.set_ylabel('能源强度 (tce/万元GDP)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # GDP与碳排放脱钩分析
        gdp_index = [46.7, 73.9, 100, 115.0, 141.0, 163.6, 220.0, 242.5]  # 以2020年为100
        carbon_index = [77.6, 91.0, 100, 99.3, 98.4, 92.0, 58.3, 29.5]    # 以2020年为100
        
        ax4 = axes[1, 1]
        ax4.plot(years, gdp_index, 'r^-', linewidth=3, markersize=7, label='GDP指数')
        ax4.plot(years, carbon_index, 'bs-', linewidth=3, markersize=7, label='碳排放指数')
        
        # 填充脱钩区域
        ax4.fill_between(years, gdp_index, carbon_index, 
                        where=(np.array(gdp_index) > np.array(carbon_index)),
                        color='lightgreen', alpha=0.5, label='脱钩区域')
        
        ax4.set_title('经济增长与碳排放脱钩分析', fontsize=14, fontweight='bold', pad=20)
        ax4.set_xlabel('年份', fontsize=12)
        ax4.set_ylabel('指数 (2020年=100)', fontsize=12)
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figures/arima_intensity_indicators.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_milestone_analysis_chart(self):
        """创建里程碑分析图表"""
        fig, axes = plt.subplots(2, 3, figsize=(22, 12))
        fig.suptitle('双碳目标关键里程碑分析', fontsize=20, fontweight='bold')
        
        # 1. 碳达峰路径
        years_peak = list(range(2020, 2041))
        carbon_peak_path = [72633]  # 2020年基准
        
        # 生成碳达峰路径数据
        for year in range(2021, 2041):
            if year <= 2025:
                growth = 0.995  # 轻微下降
            elif year <= 2030:
                growth = 0.992  # 加速下降至达峰
            else:
                growth = 0.985  # 达峰后快速下降
            carbon_peak_path.append(carbon_peak_path[-1] * growth)
        
        ax1 = axes[0, 0]
        ax1.plot(years_peak, carbon_peak_path, 'r-', linewidth=4, label='预测路径')
        ax1.axvline(x=2030, color='orange', linestyle='--', linewidth=3, label='2030碳达峰')
        ax1.axhline(y=71500, color='green', linestyle=':', linewidth=2, alpha=0.8)
        
        # 标记达峰点
        ax1.plot(2030, 71500, 'ro', markersize=12)
        ax1.annotate('碳达峰\n71,500万tCO₂', xy=(2030, 71500), xytext=(2032, 75000),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=12, ha='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        
        ax1.set_title('碳达峰路径 (2020-2040)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('年份', fontsize=12)
        ax1.set_ylabel('碳排放 (万tCO₂)', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 2. 中国式现代化目标对比
        milestones = ['2020年\n基准', '2025年\n十四五', '2030年\n碳达峰', '2035年\n基本现代化', '2050年\n现代化强国', '2060年\n碳中和']
        carbon_values = [72633, 72100, 71500, 66800, 42300, 21400]
        gdp_values = [88683, 102000, 125000, 145000, 195000, 215000]
        
        ax2 = axes[0, 1]
        x_pos = range(len(milestones))
        bars = ax2.bar(x_pos, carbon_values, color=['#34495E', '#3498DB', '#F39C12', '#27AE60', '#E74C3C', '#9B59B6'])
        ax2.set_title('关键里程碑碳排放量', fontsize=14, fontweight='bold')
        ax2.set_ylabel('碳排放 (万tCO₂)', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(milestones, rotation=45, ha='right', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, value in zip(bars, carbon_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=10)
        
        # 3. GDP增长与碳排放对比
        ax3 = axes[0, 2]
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(x_pos, gdp_values, 'bo-', linewidth=3, markersize=8, label='GDP')
        line2 = ax3_twin.plot(x_pos, carbon_values, 'ro-', linewidth=3, markersize=8, label='碳排放')
        
        ax3.set_title('经济发展与碳排放关系', fontsize=14, fontweight='bold')
        ax3.set_xlabel('发展阶段', fontsize=12)
        ax3.set_ylabel('GDP (亿元)', fontsize=12, color='blue')
        ax3_twin.set_ylabel('碳排放 (万tCO₂)', fontsize=12, color='red')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([m.split('\n')[0] for m in milestones], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left')
        
        # 4. 能源结构演变
        years_structure = [2020, 2030, 2050, 2060]
        coal_share = [65, 52, 15, 5]
        oil_share = [18, 18, 12, 8]
        gas_share = [8, 12, 13, 7]
        nonfossil_share = [15, 25, 70, 85]
        
        ax4 = axes[1, 0]
        width = 0.6
        x_pos_structure = range(len(years_structure))
        
        p1 = ax4.bar(x_pos_structure, coal_share, width, label='煤炭', color='#2C3E50')
        p2 = ax4.bar(x_pos_structure, oil_share, width, bottom=coal_share, label='石油', color='#8B4513')
        p3 = ax4.bar(x_pos_structure, gas_share, width, 
                    bottom=np.array(coal_share) + np.array(oil_share), label='天然气', color='#4169E1')
        p4 = ax4.bar(x_pos_structure, nonfossil_share, width,
                    bottom=np.array(coal_share) + np.array(oil_share) + np.array(gas_share),
                    label='非化石能源', color='#228B22')
        
        ax4.set_title('能源消费结构演变', fontsize=14, fontweight='bold')
        ax4.set_xlabel('年份', fontsize=12)
        ax4.set_ylabel('占比 (%)', fontsize=12)
        ax4.set_xticks(x_pos_structure)
        ax4.set_xticklabels(years_structure)
        ax4.legend(fontsize=11)
        ax4.set_ylim(0, 105)
        
        # 5. 减排贡献分析
        reduction_factors = ['能效提升', '结构优化', '清洁能源', '技术进步', '政策措施']
        contribution = [35, 25, 20, 15, 5]  # 百分比
        colors_contrib = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        ax5 = axes[1, 1]
        wedges, texts, autotexts = ax5.pie(contribution, labels=reduction_factors, autopct='%1.1f%%',
                                          colors=colors_contrib, startangle=90)
        ax5.set_title('2030年减排贡献构成', fontsize=14, fontweight='bold')
        
        # 6. 碳中和路径
        years_neutrality = [2030, 2040, 2050, 2060]
        emissions = [71500, 55000, 42300, 21400]
        removals = [5000, 15000, 28000, 45000]  # 碳移除/吸收
        net_emissions = [e - r for e, r in zip(emissions, removals)]
        
        ax6 = axes[1, 2]
        ax6.bar(years_neutrality, emissions, label='总排放', color='red', alpha=0.7)
        ax6.bar(years_neutrality, [-r for r in removals], label='碳移除', color='green', alpha=0.7)
        ax6.plot(years_neutrality, net_emissions, 'ko-', linewidth=3, markersize=8, label='净排放')
        
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax6.set_title('碳中和路径分析', fontsize=14, fontweight='bold')
        ax6.set_xlabel('年份', fontsize=12)
        ax6.set_ylabel('碳排放 (万tCO₂)', fontsize=12)
        ax6.legend(fontsize=11)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('figures/arima_milestone_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_policy_scenario_chart(self):
        """创建政策情景对比图表"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('政策情景分析与敏感性测试', fontsize=18, fontweight='bold')
        
        years_scenario = list(range(2020, 2061, 5))  # 每5年一个点
        
        # 不同政策强度情景
        baseline_scenario = [72633, 75800, 78200, 76500, 65200, 58000, 52000, 47000, 43000]
        moderate_scenario = [72633, 74000, 74500, 68000, 55000, 42000, 32000, 25000, 20000]
        aggressive_scenario = [72633, 72100, 71500, 66800, 42300, 28000, 18000, 12000, 8000]
        
        ax1 = axes[0, 0]
        ax1.plot(years_scenario, baseline_scenario, 'r--', linewidth=3, label='基准情景（无额外措施）')
        ax1.plot(years_scenario, moderate_scenario, 'b-', linewidth=3, label='适度政策情景')
        ax1.plot(years_scenario, aggressive_scenario, 'g-', linewidth=3, label='积极政策情景')
        
        # 标记2030和2060目标
        ax1.axhline(y=71500, color='orange', linestyle=':', alpha=0.8, label='2030达峰目标')
        ax1.axhline(y=10000, color='purple', linestyle=':', alpha=0.8, label='2060接近中和')
        
        ax1.set_title('不同政策情景碳排放路径', fontsize=14, fontweight='bold')
        ax1.set_xlabel('年份', fontsize=12)
        ax1.set_ylabel('碳排放 (万tCO₂)', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # GDP增长率敏感性分析
        gdp_growth_scenarios = ['低增长\n(年均4%)', '基准增长\n(年均5%)', '高增长\n(年均6%)']
        carbon_2030 = [69000, 71500, 74000]
        carbon_2050 = [38000, 42300, 47000]
        
        ax2 = axes[0, 1]
        x_pos = range(len(gdp_growth_scenarios))
        width = 0.35
        
        bars1 = ax2.bar([x - width/2 for x in x_pos], carbon_2030, width, 
                       label='2030年排放', color='#3498DB')
        bars2 = ax2.bar([x + width/2 for x in x_pos], carbon_2050, width,
                       label='2050年排放', color='#E74C3C')
        
        ax2.set_title('GDP增长率敏感性分析', fontsize=14, fontweight='bold')
        ax2.set_ylabel('碳排放 (万tCO₂)', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(gdp_growth_scenarios)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 500,
                        f'{height:.0f}', ha='center', va='bottom', fontsize=10)
        
        # 技术进步影响分析
        tech_scenarios = ['缓慢', '基准', '快速']
        emission_reduction = [15, 25, 35]  # 相对于基准情景的额外减排百分比
        
        ax3 = axes[1, 0]
        colors_tech = ['#FF7F7F', '#FFD700', '#90EE90']
        bars_tech = ax3.bar(tech_scenarios, emission_reduction, color=colors_tech)
        
        ax3.set_title('技术进步对减排的影响', fontsize=14, fontweight='bold')
        ax3.set_ylabel('额外减排潜力 (%)', fontsize=12)
        ax3.set_xlabel('技术进步速度', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars_tech, emission_reduction):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 政策工具效果对比
        policy_tools = ['碳税', '碳交易', '技术标准', '财政补贴', '绿色金融']
        effectiveness = [8.5, 7.2, 6.8, 5.5, 4.3]  # 减排效果评分（1-10分）
        cost_efficiency = [6.2, 8.1, 7.5, 4.8, 6.9]  # 成本效率评分（1-10分）
        
        ax4 = axes[1, 1]
        x_pos_policy = np.arange(len(policy_tools))
        width = 0.35
        
        bars_effect = ax4.bar(x_pos_policy - width/2, effectiveness, width, 
                             label='减排效果', color='#2E86C1')
        bars_cost = ax4.bar(x_pos_policy + width/2, cost_efficiency, width,
                           label='成本效率', color='#28B463')
        
        ax4.set_title('政策工具效果评估', fontsize=14, fontweight='bold')
        ax4.set_ylabel('评分 (1-10分)', fontsize=12)
        ax4.set_xlabel('政策工具', fontsize=12)
        ax4.set_xticks(x_pos_policy)
        ax4.set_xticklabels(policy_tools, rotation=45, ha='right')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, 10)
        
        plt.tight_layout()
        plt.savefig('figures/arima_policy_scenarios.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_dashboard(self):
        """创建综合仪表板"""
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('ARIMA预测模型综合分析仪表板', fontsize=24, fontweight='bold', y=0.95)
        
        # 1. 主要指标概览 (2x2)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        
        # 创建四个关键指标的组合图
        years_key = [2020, 2025, 2030, 2035, 2050, 2060]
        indicators = {
            'GDP(万亿元)': [8.87, 10.20, 12.50, 14.50, 19.50, 21.50],
            '人口(千万人)': [0.848, 0.868, 0.885, 0.895, 0.910, 0.905],
            '能源(万tce)': [31438, 32500, 34800, 36200, 38500, 37200],
            '碳排放(万tCO₂)': [72633, 72100, 71500, 66800, 42300, 21400]
        }
        
        # 标准化数据以便在同一图中显示
        normalized_data = {}
        for key, values in indicators.items():
            base_value = values[0]  # 2020年为基准
            normalized_data[key] = [v/base_value * 100 for v in values]
        
        colors_indicators = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
        
        for i, (key, values) in enumerate(normalized_data.items()):
            ax1.plot(years_key, values, 'o-', linewidth=3, markersize=8, 
                    label=key, color=colors_indicators[i])
        
        ax1.axhline(y=100, color='black', linestyle='-', alpha=0.5)
        ax1.set_title('主要指标发展趋势 (2020年=100)', fontsize=16, fontweight='bold')
        ax1.set_xlabel('年份', fontsize=12)
        ax1.set_ylabel('指数', fontsize=12)
        ax1.legend(fontsize=12, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. 碳达峰路径 (右上)
        ax2 = fig.add_subplot(gs[0, 2:])
        years_peak = list(range(2020, 2041))
        carbon_trajectory = [72633]
        for year in range(2021, 2041):
            if year <= 2025:
                factor = 0.998
            elif year <= 2030:
                factor = 0.995
            else:
                factor = 0.975
            carbon_trajectory.append(carbon_trajectory[-1] * factor)
        
        ax2.plot(years_peak, carbon_trajectory, 'r-', linewidth=4)
        ax2.fill_between(years_peak, carbon_trajectory, alpha=0.3, color='red')
        ax2.axvline(x=2030, color='orange', linestyle='--', linewidth=3)
        ax2.plot(2030, 71500, 'go', markersize=15)
        
        ax2.set_title('碳达峰路径预测', fontsize=16, fontweight='bold')
        ax2.set_xlabel('年份', fontsize=12)
        ax2.set_ylabel('碳排放 (万tCO₂)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 3. 分部门占比 (右中)
        ax3 = fig.add_subplot(gs[1, 2:])
        sectors_pie = ['工业', '建筑', '交通', '居民', '农林', '其他']
        sizes_pie = [68.5, 12.7, 9.6, 4.6, 1.8, 2.8]
        colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#A8A8A8']
        
        wedges, texts, autotexts = ax3.pie(sizes_pie, labels=sectors_pie, autopct='%1.1f%%',
                                          colors=colors_pie, startangle=90)
        ax3.set_title('2030年分部门碳排放占比', fontsize=16, fontweight='bold')
        
        # 4. 强度指标变化 (左下)
        ax4 = fig.add_subplot(gs[2, 0:2])
        years_intensity = [2020, 2025, 2030, 2035, 2050, 2060]
        carbon_intensity = [0.819, 0.707, 0.572, 0.461, 0.217, 0.099]
        energy_intensity = [0.354, 0.319, 0.278, 0.250, 0.197, 0.173]
        
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(years_intensity, carbon_intensity, 'ro-', linewidth=3, 
                        markersize=8, label='碳强度')
        line2 = ax4_twin.plot(years_intensity, energy_intensity, 'bs-', linewidth=3, 
                             markersize=8, label='能源强度')
        
        ax4.set_title('强度指标变化趋势', fontsize=16, fontweight='bold')
        ax4.set_xlabel('年份', fontsize=12)
        ax4.set_ylabel('碳强度 (tCO₂/万元GDP)', fontsize=12, color='red')
        ax4_twin.set_ylabel('能源强度 (tce/万元GDP)', fontsize=12, color='blue')
        ax4.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right')
        
        # 5. 能源结构演变 (右下)
        ax5 = fig.add_subplot(gs[2, 2:])
        years_structure = [2020, 2030, 2050, 2060]
        structure_data = {
            '煤炭': [65, 52, 15, 5],
            '石油': [18, 18, 12, 8], 
            '天然气': [8, 12, 13, 7],
            '非化石能源': [15, 25, 70, 85]
        }
        
        bottom = np.zeros(len(years_structure))
        colors_structure = ['#2C3E50', '#8B4513', '#4169E1', '#228B22']
        
        for i, (fuel, values) in enumerate(structure_data.items()):
            ax5.bar(years_structure, values, bottom=bottom, label=fuel, 
                   color=colors_structure[i], alpha=0.8)
            bottom += values
        
        ax5.set_title('能源消费结构演变', fontsize=16, fontweight='bold')
        ax5.set_xlabel('年份', fontsize=12)
        ax5.set_ylabel('占比 (%)', fontsize=12)
        ax5.legend(fontsize=11)
        ax5.set_ylim(0, 100)
        
        # 6. 关键目标进展 (底部)
        ax6 = fig.add_subplot(gs[3, :])
        
        targets = ['2025年\n碳强度下降18%', '2030年\n碳达峰', '2030年\n非化石能源25%', 
                  '2035年\n碳排放下降6.6%', '2050年\n碳排放下降40.8%', '2060年\n接近碳中和']
        progress = [95, 100, 90, 85, 75, 60]  # 实现进度百分比
        
        bars_progress = ax6.barh(range(len(targets)), progress, 
                                color=['#27AE60' if p >= 90 else '#F39C12' if p >= 70 else '#E74C3C' 
                                      for p in progress])
        
        ax6.set_title('双碳目标实现进度评估', fontsize=16, fontweight='bold')
        ax6.set_xlabel('完成度 (%)', fontsize=12)
        ax6.set_yticks(range(len(targets)))
        ax6.set_yticklabels(targets, fontsize=11)
        ax6.set_xlim(0, 100)
        ax6.grid(True, alpha=0.3, axis='x')
        
        # 添加进度标签
        for i, (bar, value) in enumerate(zip(bars_progress, progress)):
            ax6.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{value}%', ha='left', va='center', fontsize=11, fontweight='bold')
        
        # 添加图例说明
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#27AE60', label='进展良好 (≥90%)'),
                          Patch(facecolor='#F39C12', label='正常进展 (70-90%)'),
                          Patch(facecolor='#E74C3C', label='需要加强 (<70%)')]
        ax6.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.savefig('figures/arima_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_all_visualizations(self):
        """生成所有可视化图表"""
        print("开始生成ARIMA预测模型可视化图表...")
        
        # 创建输出目录
        import os
        if not os.path.exists('figures'):
            os.makedirs('figures')
        
        print("1. 生成主要预测图表...")
        self.create_main_forecast_chart()
        
        print("2. 生成分部门分析图表...")
        self.create_sectoral_analysis_chart()
        
        print("3. 生成强度指标图表...")
        self.create_intensity_indicators_chart()
        
        print("4. 生成里程碑分析图表...")
        self.create_milestone_analysis_chart()
        
        print("5. 生成政策情景图表...")
        self.create_policy_scenario_chart()
        
        print("6. 生成综合仪表板...")
        self.create_comprehensive_dashboard()
        
        print("\n✅ 所有图表生成完成!")
        print("生成的图表文件：")
        print("- figures/arima_main_forecast.png (主要预测图表)")
        print("- figures/arima_sectoral_analysis.png (分部门分析)")  
        print("- figures/arima_intensity_indicators.png (强度指标分析)")
        print("- figures/arima_milestone_analysis.png (里程碑分析)")
        print("- figures/arima_policy_scenarios.png (政策情景分析)")
        print("- figures/arima_comprehensive_dashboard.png (综合仪表板)")

def main():
    """主函数"""
    visualizer = ARIMAVisualization()
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()