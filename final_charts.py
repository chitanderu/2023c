#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完成剩余图表生成
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 强度指标图表
print("Generating intensity indicators chart...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Carbon Emission Intensity & Per Capita Analysis', fontsize=18, fontweight='bold')

years = [2010, 2015, 2020, 2025, 2030, 2035, 2050, 2060]
carbon_intensity = [1.362, 1.008, 0.819, 0.707, 0.572, 0.461, 0.217, 0.099]
per_capita = [7.14, 7.95, 8.57, 8.31, 8.08, 7.47, 4.65, 2.36]
energy_intensity = [0.568, 0.443, 0.354, 0.319, 0.278, 0.250, 0.197, 0.173]

axes[0,0].plot(years, carbon_intensity, 'ro-', linewidth=4, markersize=8)
axes[0,0].fill_between(years, carbon_intensity, alpha=0.3, color='red')
axes[0,0].set_title('Carbon Intensity Trend')
axes[0,0].set_ylabel('tCO2/10k Yuan GDP')
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(years, per_capita, 'go-', linewidth=4, markersize=8)
axes[0,1].set_title('Per Capita Carbon Emissions')
axes[0,1].set_ylabel('tCO2/person')
axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(years, energy_intensity, 'bo-', linewidth=4, markersize=8)
axes[1,0].set_title('Energy Intensity Trend')
axes[1,0].set_ylabel('tce/10k Yuan GDP')
axes[1,0].grid(True, alpha=0.3)

gdp_index = [46.7, 73.9, 100, 115.0, 141.0, 163.6, 220.0, 242.5]
carbon_index = [77.6, 91.0, 100, 99.3, 98.4, 92.0, 58.3, 29.5]
axes[1,1].plot(years, gdp_index, 'r^-', linewidth=3, label='GDP Index')
axes[1,1].plot(years, carbon_index, 'bs-', linewidth=3, label='Carbon Index')
axes[1,1].set_title('GDP vs Carbon Decoupling')
axes[1,1].set_ylabel('Index (2020=100)')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/arima_intensity_indicators.png', dpi=300, bbox_inches='tight')
plt.close()
print("Intensity indicators chart saved.")

# 里程碑分析图表
print("Generating milestone analysis chart...")
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Carbon Peak & Neutrality Milestone Analysis', fontsize=20, fontweight='bold')

# 碳达峰路径
years_peak = list(range(2020, 2041))
carbon_peak = [72633 * (0.998 ** (i-2020)) if i <= 2025 else 
               72633 * 0.99 * (0.995 ** (i-2025)) if i <= 2030 else 
               71500 * (0.985 ** (i-2030)) for i in years_peak]

axes[0,0].plot(years_peak, carbon_peak, 'r-', linewidth=4)
axes[0,0].axvline(x=2030, color='orange', linestyle='--', linewidth=3)
axes[0,0].plot(2030, 71500, 'ro', markersize=12)
axes[0,0].set_title('Carbon Peak Pathway (2020-2040)')
axes[0,0].set_ylabel('Carbon Emissions (10k tCO2)')
axes[0,0].grid(True, alpha=0.3)

# 关键里程碑
milestones = ['2020', '2025', '2030', '2035', '2050', '2060']
carbon_values = [72633, 72100, 71500, 66800, 42300, 21400]
colors = ['#34495E', '#3498DB', '#F39C12', '#27AE60', '#E74C3C', '#9B59B6']

axes[0,1].bar(range(len(milestones)), carbon_values, color=colors)
axes[0,1].set_title('Key Milestone Emissions')
axes[0,1].set_ylabel('Carbon Emissions (10k tCO2)')
axes[0,1].set_xticks(range(len(milestones)))
axes[0,1].set_xticklabels(milestones)

# GDP vs 碳排放
gdp_values = [88683, 102000, 125000, 145000, 195000, 215000]
ax_twin = axes[0,2].twinx()
axes[0,2].plot(range(len(milestones)), gdp_values, 'bo-', linewidth=3, label='GDP')
ax_twin.plot(range(len(milestones)), carbon_values, 'ro-', linewidth=3, label='Carbon')
axes[0,2].set_title('GDP vs Carbon Relationship')
axes[0,2].set_ylabel('GDP (billion yuan)', color='blue')
ax_twin.set_ylabel('Carbon (10k tCO2)', color='red')

# 能源结构
years_struct = [2020, 2030, 2050, 2060]
coal = [65, 52, 15, 5]
oil = [18, 18, 12, 8]
gas = [8, 12, 13, 7]
nonfossil = [15, 25, 70, 85]

width = 0.6
axes[1,0].bar(years_struct, coal, width, label='Coal', color='#2C3E50')
axes[1,0].bar(years_struct, oil, width, bottom=coal, label='Oil', color='#8B4513')
axes[1,0].bar(years_struct, gas, width, bottom=np.array(coal)+np.array(oil), label='Gas', color='#4169E1')
axes[1,0].bar(years_struct, nonfossil, width, 
             bottom=np.array(coal)+np.array(oil)+np.array(gas), label='Non-fossil', color='#228B22')
axes[1,0].set_title('Energy Structure Evolution')
axes[1,0].set_ylabel('Share (%)')
axes[1,0].legend()

# 减排贡献
factors = ['Energy\nEfficiency', 'Structure\nOptimization', 'Clean\nEnergy', 'Technology\nProgress', 'Policy\nMeasures']
contrib = [35, 25, 20, 15, 5]
colors_contrib = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']

axes[1,1].pie(contrib, labels=factors, autopct='%1.1f%%', colors=colors_contrib, startangle=90)
axes[1,1].set_title('2030 Emission Reduction Contributions')

# 碳中和路径
years_neut = [2030, 2040, 2050, 2060]
emissions = [71500, 55000, 42300, 21400]
removals = [5000, 15000, 28000, 45000]
net = [e - r for e, r in zip(emissions, removals)]

axes[1,2].bar(years_neut, emissions, label='Total Emissions', color='red', alpha=0.7)
axes[1,2].bar(years_neut, [-r for r in removals], label='Carbon Removal', color='green', alpha=0.7)
axes[1,2].plot(years_neut, net, 'ko-', linewidth=3, markersize=8, label='Net Emissions')
axes[1,2].axhline(y=0, color='black', linestyle='-', linewidth=2)
axes[1,2].set_title('Carbon Neutrality Pathway')
axes[1,2].set_ylabel('Carbon Emissions (10k tCO2)')
axes[1,2].legend()

plt.tight_layout()
plt.savefig('figures/arima_milestone_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("Milestone analysis chart saved.")

# 政策情景分析
print("Generating policy scenario chart...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Policy Scenario Analysis & Sensitivity Test', fontsize=18, fontweight='bold')

years_scenario = [2020, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060]
baseline = [72633, 75800, 78200, 76500, 65200, 58000, 52000, 47000, 43000]
moderate = [72633, 74000, 74500, 68000, 55000, 42000, 32000, 25000, 20000]
aggressive = [72633, 72100, 71500, 66800, 42300, 28000, 18000, 12000, 8000]

axes[0,0].plot(years_scenario, baseline, 'r--', linewidth=3, label='Baseline Scenario')
axes[0,0].plot(years_scenario, moderate, 'b-', linewidth=3, label='Moderate Policy')
axes[0,0].plot(years_scenario, aggressive, 'g-', linewidth=3, label='Aggressive Policy')
axes[0,0].set_title('Policy Scenario Comparison')
axes[0,0].set_ylabel('Carbon Emissions (10k tCO2)')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# GDP敏感性
scenarios = ['Low\nGrowth', 'Baseline', 'High\nGrowth']
carbon_2030 = [69000, 71500, 74000]
carbon_2050 = [38000, 42300, 47000]

x_pos = range(len(scenarios))
width = 0.35
axes[0,1].bar([x - width/2 for x in x_pos], carbon_2030, width, label='2030', color='#3498DB')
axes[0,1].bar([x + width/2 for x in x_pos], carbon_2050, width, label='2050', color='#E74C3C')
axes[0,1].set_title('GDP Growth Rate Sensitivity')
axes[0,1].set_ylabel('Carbon Emissions (10k tCO2)')
axes[0,1].set_xticks(x_pos)
axes[0,1].set_xticklabels(scenarios)
axes[0,1].legend()

# 技术进步
tech_scenarios = ['Slow', 'Baseline', 'Fast']
reduction = [15, 25, 35]
colors_tech = ['#FF7F7F', '#FFD700', '#90EE90']

axes[1,0].bar(tech_scenarios, reduction, color=colors_tech)
axes[1,0].set_title('Technology Progress Impact')
axes[1,0].set_ylabel('Additional Reduction Potential (%)')
for i, v in enumerate(reduction):
    axes[1,0].text(i, v + 1, f'{v}%', ha='center', fontweight='bold')

# 政策工具效果
tools = ['Carbon\nTax', 'ETS', 'Standards', 'Subsidies', 'Green\nFinance']
effectiveness = [8.5, 7.2, 6.8, 5.5, 4.3]
cost_efficiency = [6.2, 8.1, 7.5, 4.8, 6.9]

x_pos_policy = np.arange(len(tools))
width = 0.35
axes[1,1].bar(x_pos_policy - width/2, effectiveness, width, label='Effectiveness', color='#2E86C1')
axes[1,1].bar(x_pos_policy + width/2, cost_efficiency, width, label='Cost Efficiency', color='#28B463')
axes[1,1].set_title('Policy Tool Assessment')
axes[1,1].set_ylabel('Score (1-10)')
axes[1,1].set_xticks(x_pos_policy)
axes[1,1].set_xticklabels(tools)
axes[1,1].legend()
axes[1,1].set_ylim(0, 10)

plt.tight_layout()
plt.savefig('figures/arima_policy_scenarios.png', dpi=300, bbox_inches='tight')
plt.close()
print("Policy scenario chart saved.")

# 综合仪表板
print("Generating comprehensive dashboard...")
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
fig.suptitle('ARIMA Forecasting Model Comprehensive Dashboard', fontsize=20, fontweight='bold', y=0.95)

# 主要指标趋势
ax1 = fig.add_subplot(gs[0:2, 0:2])
years_key = [2020, 2025, 2030, 2035, 2050, 2060]
gdp_norm = [100, 115, 141, 164, 220, 242]
pop_norm = [100, 102, 104, 106, 107, 107]
energy_norm = [100, 103, 111, 115, 122, 118]
carbon_norm = [100, 99, 98, 92, 58, 29]

ax1.plot(years_key, gdp_norm, 'ro-', linewidth=3, markersize=8, label='GDP')
ax1.plot(years_key, pop_norm, 'bs-', linewidth=3, markersize=8, label='Population')
ax1.plot(years_key, energy_norm, 'g^-', linewidth=3, markersize=8, label='Energy')
ax1.plot(years_key, carbon_norm, 'mo-', linewidth=3, markersize=8, label='Carbon')

ax1.axhline(y=100, color='black', linestyle='-', alpha=0.5)
ax1.set_title('Key Indicators Trend (2020=100)', fontsize=16, fontweight='bold')
ax1.set_ylabel('Index')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 碳达峰路径
ax2 = fig.add_subplot(gs[0, 2:])
peak_path = [72633 * (0.998 ** (i-2020)) if i <= 2025 else 
             72633 * 0.99 * (0.995 ** (i-2025)) if i <= 2030 else 
             71500 * (0.985 ** (i-2030)) for i in years_peak]

ax2.plot(years_peak, peak_path, 'r-', linewidth=4)
ax2.fill_between(years_peak, peak_path, alpha=0.3, color='red')
ax2.axvline(x=2030, color='orange', linestyle='--', linewidth=3)
ax2.plot(2030, 71500, 'go', markersize=15)
ax2.set_title('Carbon Peak Pathway Forecast', fontsize=16, fontweight='bold')
ax2.set_ylabel('Carbon Emissions (10k tCO2)')
ax2.grid(True, alpha=0.3)

# 分部门占比
ax3 = fig.add_subplot(gs[1, 2:])
sectors = ['Industry', 'Building', 'Transport', 'Residential', 'Agriculture', 'Others']
sizes = [68.5, 12.7, 9.6, 4.6, 1.8, 2.8]
colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#A8A8A8']

ax3.pie(sizes, labels=sectors, autopct='%1.1f%%', colors=colors_pie, startangle=90)
ax3.set_title('2030 Sectoral Carbon Emissions', fontsize=16, fontweight='bold')

# 能源结构演变
ax4 = fig.add_subplot(gs[2, :])
years_struct = [2020, 2030, 2050, 2060]
coal_data = [65, 52, 15, 5]
oil_data = [18, 18, 12, 8]
gas_data = [8, 12, 13, 7]
nonfossil_data = [15, 25, 70, 85]

width = 0.6
ax4.bar(years_struct, coal_data, width, label='Coal', color='#2C3E50', alpha=0.8)
ax4.bar(years_struct, oil_data, width, bottom=coal_data, label='Oil', color='#8B4513', alpha=0.8)
ax4.bar(years_struct, gas_data, width, bottom=np.array(coal_data)+np.array(oil_data), 
        label='Natural Gas', color='#4169E1', alpha=0.8)
ax4.bar(years_struct, nonfossil_data, width, 
       bottom=np.array(coal_data)+np.array(oil_data)+np.array(gas_data), 
       label='Non-fossil Energy', color='#228B22', alpha=0.8)

ax4.set_title('Energy Structure Evolution Trend', fontsize=16, fontweight='bold')
ax4.set_xlabel('Year')
ax4.set_ylabel('Share (%)')
ax4.legend(loc='center right', bbox_to_anchor=(1.15, 0.5))
ax4.set_ylim(0, 100)

plt.savefig('figures/arima_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("Comprehensive dashboard saved.")

print("\nAll ARIMA visualization charts have been generated successfully!")
print("Generated files:")
print("- arima_main_forecast.png")
print("- arima_sectoral_analysis.png") 
print("- arima_intensity_indicators.png")
print("- arima_milestone_analysis.png")
print("- arima_policy_scenarios.png")
print("- arima_comprehensive_dashboard.png")