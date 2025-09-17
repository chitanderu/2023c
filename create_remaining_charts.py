#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速生成剩余的ARIMA预测图表
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_sectoral_chart():
    """创建分部门分析图表"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('分部门能源消费与碳排放分析', fontsize=18, fontweight='bold')
    
    # 2030年分部门能源消费占比
    sectors = ['工业部门', '建筑部门', '交通部门', '居民生活', '农林部门']
    shares = [70.0, 14.0, 9.0, 5.0, 2.0]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    axes[0,0].pie(shares, labels=sectors, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0,0].set_title('2030年分部门能源消费占比', fontsize=14, fontweight='bold')
    
    # 分部门能源消费趋势
    years = [2020, 2025, 2030, 2035, 2050]
    industry = [23625, 24150, 24360, 25020, 26950]
    building = [3774, 4225, 4872, 5330, 6160]
    transport = [2514, 2795, 3132, 3258, 3465]
    
    axes[0,1].plot(years, industry, 'o-', label='工业部门', linewidth=3, color='#FF6B6B')
    axes[0,1].plot(years, building, 's-', label='建筑部门', linewidth=3, color='#4ECDC4')
    axes[0,1].plot(years, transport, '^-', label='交通部门', linewidth=3, color='#45B7D1')
    axes[0,1].set_title('主要部门能源消费趋势', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('年份')
    axes[0,1].set_ylabel('能源消费 (万tce)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 分部门排放因子变化
    factors_years = [2020, 2030, 2050]
    factors = {
        '工业部门': [2.8, 2.01, 1.2],
        '建筑部门': [2.2, 1.86, 1.0],
        '交通部门': [2.4, 2.20, 0.8],
        '居民生活': [2.0, 1.90, 1.1]
    }
    
    for sector, values in factors.items():
        axes[1,0].plot(factors_years, values, 'o-', label=sector, linewidth=2.5, markersize=6)
    
    axes[1,0].set_title('分部门碳排放因子变化', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('年份')
    axes[1,0].set_ylabel('排放因子 (tCO₂/tce)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 2030年分部门碳排放量
    carbon_sectors = sectors + ['能源供应']
    carbon_values = [48960, 9062, 6889, 3306, 1283, 2000]
    colors_carbon = colors + ['#A8A8A8']
    
    bars = axes[1,1].bar(range(len(carbon_sectors)), carbon_values, color=colors_carbon)
    axes[1,1].set_title('2030年分部门碳排放量', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('部门')
    axes[1,1].set_ylabel('碳排放 (万tCO₂)')
    axes[1,1].set_xticks(range(len(carbon_sectors)))
    axes[1,1].set_xticklabels(carbon_sectors, rotation=45, ha='right')
    axes[1,1].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, value in zip(bars, carbon_values):
        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                      f'{value:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/arima_sectoral_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_intensity_chart():
    """创建强度指标图表"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('碳排放强度与人均指标分析', fontsize=18, fontweight='bold')
    
    years = [2010, 2015, 2020, 2025, 2030, 2035, 2050, 2060]
    
    # 碳排放强度
    carbon_intensity = [1.362, 1.008, 0.819, 0.707, 0.572, 0.461, 0.217, 0.099]
    
    axes[0,0].plot(years, carbon_intensity, 'ro-', linewidth=4, markersize=8)
    axes[0,0].fill_between(years, carbon_intensity, alpha=0.3, color='red')
    axes[0,0].axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='2030年目标')
    axes[0,0].axhline(y=0.2, color='green', linestyle='--', linewidth=2, label='2050年目标')
    axes[0,0].set_title('碳排放强度变化趋势', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('年份')
    axes[0,0].set_ylabel('碳强度 (tCO₂/万元GDP)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 人均碳排放
    per_capita = [7.14, 7.95, 8.57, 8.31, 8.08, 7.47, 4.65, 2.36]
    
    axes[0,1].plot(years, per_capita, 'go-', linewidth=4, markersize=8)
    axes[0,1].fill_between(years, per_capita, alpha=0.3, color='green')
    axes[0,1].axhline(y=6.0, color='blue', linestyle='--', label='发达国家平均')
    axes[0,1].axhline(y=3.0, color='purple', linestyle='--', label='欧盟当前水平')
    axes[0,1].set_title('人均碳排放变化趋势', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('年份')
    axes[0,1].set_ylabel('人均碳排放 (tCO₂/人)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 能源强度
    energy_intensity = [0.568, 0.443, 0.354, 0.319, 0.278, 0.250, 0.197, 0.173]
    
    axes[1,0].plot(years, energy_intensity, 'bo-', linewidth=4, markersize=8)
    axes[1,0].fill_between(years, energy_intensity, alpha=0.3, color='blue')
    axes[1,0].set_title('能源消费强度变化趋势', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('年份')
    axes[1,0].set_ylabel('能源强度 (tce/万元GDP)')
    axes[1,0].grid(True, alpha=0.3)
    
    # GDP与碳排放脱钩
    gdp_index = [46.7, 73.9, 100, 115.0, 141.0, 163.6, 220.0, 242.5]
    carbon_index = [77.6, 91.0, 100, 99.3, 98.4, 92.0, 58.3, 29.5]
    
    axes[1,1].plot(years, gdp_index, 'r^-', linewidth=3, markersize=7, label='GDP指数')
    axes[1,1].plot(years, carbon_index, 'bs-', linewidth=3, markersize=7, label='碳排放指数')
    axes[1,1].fill_between(years, gdp_index, carbon_index, 
                          where=(np.array(gdp_index) > np.array(carbon_index)),
                          color='lightgreen', alpha=0.5, label='脱钩区域')
    axes[1,1].set_title('经济增长与碳排放脱钩分析', fontsize=14, fontweight='bold')
    axes[1,1].set_xlabel('年份')
    axes[1,1].set_ylabel('指数 (2020年=100)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/arima_intensity_indicators.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_milestone_chart():
    """创建里程碑分析图表"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('双碳目标关键里程碑分析', fontsize=20, fontweight='bold')
    
    # 1. 碳达峰路径
    years_peak = list(range(2020, 2041))
    carbon_peak = [72633 * (0.995 ** (i-2020)) if i <= 2030 else 71500 * (0.985 ** (i-2030)) 
                   for i in years_peak]
    
    axes[0,0].plot(years_peak, carbon_peak, 'r-', linewidth=4, label='预测路径')
    axes[0,0].axvline(x=2030, color='orange', linestyle='--', linewidth=3, label='2030碳达峰')
    axes[0,0].plot(2030, 71500, 'ro', markersize=12)
    axes[0,0].annotate('碳达峰\n71,500万tCO₂', xy=(2030, 71500), xytext=(2032, 75000),
                      arrowprops=dict(arrowstyle='->', color='red', lw=2))
    axes[0,0].set_title('碳达峰路径 (2020-2040)', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('年份')
    axes[0,0].set_ylabel('碳排放 (万tCO₂)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 关键里程碑对比
    milestones = ['2020基准', '2025十四五', '2030达峰', '2035现代化', '2050强国', '2060中和']
    carbon_values = [72633, 72100, 71500, 66800, 42300, 21400]
    colors_mile = ['#34495E', '#3498DB', '#F39C12', '#27AE60', '#E74C3C', '#9B59B6']
    
    bars = axes[0,1].bar(range(len(milestones)), carbon_values, color=colors_mile)
    axes[0,1].set_title('关键里程碑碳排放量', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('碳排放 (万tCO₂)')
    axes[0,1].set_xticks(range(len(milestones)))
    axes[0,1].set_xticklabels(milestones, rotation=45, ha='right')
    axes[0,1].grid(True, alpha=0.3, axis='y')
    
    # 3. GDP与碳排放关系
    gdp_values = [88683, 102000, 125000, 145000, 195000, 215000]
    
    ax_twin = axes[0,2].twinx()
    line1 = axes[0,2].plot(range(len(milestones)), gdp_values, 'bo-', linewidth=3, label='GDP')
    line2 = ax_twin.plot(range(len(milestones)), carbon_values, 'ro-', linewidth=3, label='碳排放')
    
    axes[0,2].set_title('经济发展与碳排放关系', fontsize=14, fontweight='bold')
    axes[0,2].set_xlabel('发展阶段')
    axes[0,2].set_ylabel('GDP (亿元)', color='blue')
    ax_twin.set_ylabel('碳排放 (万tCO₂)', color='red')
    axes[0,2].set_xticks(range(len(milestones)))
    axes[0,2].set_xticklabels([m.split('基准')[0] if '基准' in m else m.split('十')[0] if '十' in m 
                              else m for m in milestones], rotation=45, ha='right')
    
    # 4. 能源结构演变
    years_struct = [2020, 2030, 2050, 2060]
    coal = [65, 52, 15, 5]
    oil = [18, 18, 12, 8]
    gas = [8, 12, 13, 7]
    nonfossil = [15, 25, 70, 85]
    
    width = 0.6
    axes[1,0].bar(years_struct, coal, width, label='煤炭', color='#2C3E50')
    axes[1,0].bar(years_struct, oil, width, bottom=coal, label='石油', color='#8B4513')
    axes[1,0].bar(years_struct, gas, width, bottom=np.array(coal)+np.array(oil), label='天然气', color='#4169E1')
    axes[1,0].bar(years_struct, nonfossil, width, 
                 bottom=np.array(coal)+np.array(oil)+np.array(gas), label='非化石能源', color='#228B22')
    
    axes[1,0].set_title('能源消费结构演变', fontsize=14, fontweight='bold')
    axes[1,0].set_xlabel('年份')
    axes[1,0].set_ylabel('占比 (%)')
    axes[1,0].legend()
    axes[1,0].set_ylim(0, 105)
    
    # 5. 减排贡献
    factors = ['能效提升', '结构优化', '清洁能源', '技术进步', '政策措施']
    contrib = [35, 25, 20, 15, 5]
    colors_contrib = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    axes[1,1].pie(contrib, labels=factors, autopct='%1.1f%%', colors=colors_contrib, startangle=90)
    axes[1,1].set_title('2030年减排贡献构成', fontsize=14, fontweight='bold')
    
    # 6. 碳中和路径
    years_neut = [2030, 2040, 2050, 2060]
    emissions = [71500, 55000, 42300, 21400]
    removals = [5000, 15000, 28000, 45000]
    net_emissions = [e - r for e, r in zip(emissions, removals)]
    
    axes[1,2].bar(years_neut, emissions, label='总排放', color='red', alpha=0.7)
    axes[1,2].bar(years_neut, [-r for r in removals], label='碳移除', color='green', alpha=0.7)
    axes[1,2].plot(years_neut, net_emissions, 'ko-', linewidth=3, markersize=8, label='净排放')
    axes[1,2].axhline(y=0, color='black', linestyle='-', linewidth=2)
    axes[1,2].set_title('碳中和路径分析', fontsize=14, fontweight='bold')
    axes[1,2].set_xlabel('年份')
    axes[1,2].set_ylabel('碳排放 (万tCO₂)')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/arima_milestone_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_policy_scenarios():
    """创建政策情景对比"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('政策情景分析与敏感性测试', fontsize=18, fontweight='bold')
    
    years_scenario = [2020, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060]
    
    # 不同情景
    baseline = [72633, 75800, 78200, 76500, 65200, 58000, 52000, 47000, 43000]
    moderate = [72633, 74000, 74500, 68000, 55000, 42000, 32000, 25000, 20000]
    aggressive = [72633, 72100, 71500, 66800, 42300, 28000, 18000, 12000, 8000]
    
    axes[0,0].plot(years_scenario, baseline, 'r--', linewidth=3, label='基准情景')
    axes[0,0].plot(years_scenario, moderate, 'b-', linewidth=3, label='适度政策')
    axes[0,0].plot(years_scenario, aggressive, 'g-', linewidth=3, label='积极政策')
    axes[0,0].axhline(y=71500, color='orange', linestyle=':', alpha=0.8, label='2030达峰目标')
    axes[0,0].set_title('不同政策情景碳排放路径', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('年份')
    axes[0,0].set_ylabel('碳排放 (万tCO₂)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # GDP增长敏感性
    scenarios = ['低增长\n(4%)', '基准\n(5%)', '高增长\n(6%)']
    carbon_2030 = [69000, 71500, 74000]
    carbon_2050 = [38000, 42300, 47000]
    
    x_pos = range(len(scenarios))
    width = 0.35
    
    bars1 = axes[0,1].bar([x - width/2 for x in x_pos], carbon_2030, width, 
                         label='2030年排放', color='#3498DB')
    bars2 = axes[0,1].bar([x + width/2 for x in x_pos], carbon_2050, width,
                         label='2050年排放', color='#E74C3C')
    
    axes[0,1].set_title('GDP增长率敏感性分析', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('碳排放 (万tCO₂)')
    axes[0,1].set_xticks(x_pos)
    axes[0,1].set_xticklabels(scenarios)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3, axis='y')
    
    # 技术进步影响
    tech_scenarios = ['缓慢', '基准', '快速']
    reduction = [15, 25, 35]
    colors_tech = ['#FF7F7F', '#FFD700', '#90EE90']
    
    bars_tech = axes[1,0].bar(tech_scenarios, reduction, color=colors_tech)
    axes[1,0].set_title('技术进步对减排的影响', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('额外减排潜力 (%)')
    axes[1,0].set_xlabel('技术进步速度')
    axes[1,0].grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars_tech, reduction):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                      f'{value}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 政策工具效果
    tools = ['碳税', '碳交易', '技术标准', '财政补贴', '绿色金融']
    effectiveness = [8.5, 7.2, 6.8, 5.5, 4.3]
    cost_efficiency = [6.2, 8.1, 7.5, 4.8, 6.9]
    
    x_pos_policy = np.arange(len(tools))
    width = 0.35
    
    bars_eff = axes[1,1].bar(x_pos_policy - width/2, effectiveness, width, 
                            label='减排效果', color='#2E86C1')
    bars_cost = axes[1,1].bar(x_pos_policy + width/2, cost_efficiency, width,
                             label='成本效率', color='#28B463')
    
    axes[1,1].set_title('政策工具效果评估', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('评分 (1-10分)')
    axes[1,1].set_xlabel('政策工具')
    axes[1,1].set_xticks(x_pos_policy)
    axes[1,1].set_xticklabels(tools, rotation=45, ha='right')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3, axis='y')
    axes[1,1].set_ylim(0, 10)
    
    plt.tight_layout()
    plt.savefig('figures/arima_policy_scenarios.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_dashboard():
    """创建综合仪表板"""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    fig.suptitle('ARIMA预测模型综合分析仪表板', fontsize=20, fontweight='bold', y=0.95)
    
    # 1. 主要指标趋势 (左上，2x2)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    years_key = [2020, 2025, 2030, 2035, 2050, 2060]
    
    # 标准化数据
    gdp_norm = [100, 115, 141, 164, 220, 242]
    pop_norm = [100, 102, 104, 106, 107, 107]
    energy_norm = [100, 103, 111, 115, 122, 118]
    carbon_norm = [100, 99, 98, 92, 58, 29]
    
    ax1.plot(years_key, gdp_norm, 'ro-', linewidth=3, markersize=8, label='GDP')
    ax1.plot(years_key, pop_norm, 'bs-', linewidth=3, markersize=8, label='人口')
    ax1.plot(years_key, energy_norm, 'g^-', linewidth=3, markersize=8, label='能源消费')
    ax1.plot(years_key, carbon_norm, 'mo-', linewidth=3, markersize=8, label='碳排放')
    
    ax1.axhline(y=100, color='black', linestyle='-', alpha=0.5)
    ax1.set_title('主要指标发展趋势 (2020年=100)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('年份')
    ax1.set_ylabel('指数')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. 碳达峰路径 (右上)
    ax2 = fig.add_subplot(gs[0, 2:])
    years_peak = list(range(2020, 2041))
    peak_path = [72633 * (0.998 ** (i-2020)) if i <= 2025 else 
                 72633 * 0.99 * (0.995 ** (i-2025)) if i <= 2030 else 
                 71500 * (0.985 ** (i-2030)) for i in years_peak]
    
    ax2.plot(years_peak, peak_path, 'r-', linewidth=4)
    ax2.fill_between(years_peak, peak_path, alpha=0.3, color='red')
    ax2.axvline(x=2030, color='orange', linestyle='--', linewidth=3)
    ax2.plot(2030, 71500, 'go', markersize=15)
    ax2.set_title('碳达峰路径预测', fontsize=16, fontweight='bold')
    ax2.set_xlabel('年份')
    ax2.set_ylabel('碳排放 (万tCO₂)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 分部门占比 (右中)
    ax3 = fig.add_subplot(gs[1, 2:])
    sectors = ['工业', '建筑', '交通', '居民', '农林', '其他']
    sizes = [68.5, 12.7, 9.6, 4.6, 1.8, 2.8]
    colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#A8A8A8']
    
    ax3.pie(sizes, labels=sectors, autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax3.set_title('2030年分部门碳排放占比', fontsize=16, fontweight='bold')
    
    # 4. 能源结构演变 (底部全幅)
    ax4 = fig.add_subplot(gs[2, :])
    years_struct = [2020, 2030, 2050, 2060]
    
    coal_data = [65, 52, 15, 5]
    oil_data = [18, 18, 12, 8]
    gas_data = [8, 12, 13, 7]
    nonfossil_data = [15, 25, 70, 85]
    
    width = 0.6
    bottom1 = np.array([0] * len(years_struct))
    bottom2 = np.array(coal_data)
    bottom3 = bottom2 + np.array(oil_data)
    
    ax4.bar(years_struct, coal_data, width, label='煤炭', color='#2C3E50', alpha=0.8)
    ax4.bar(years_struct, oil_data, width, bottom=bottom2, label='石油', color='#8B4513', alpha=0.8)
    ax4.bar(years_struct, gas_data, width, bottom=bottom3, label='天然气', color='#4169E1', alpha=0.8)
    ax4.bar(years_struct, nonfossil_data, width, 
           bottom=bottom3 + np.array(gas_data), label='非化石能源', color='#228B22', alpha=0.8)
    
    ax4.set_title('能源消费结构演变趋势', fontsize=16, fontweight='bold')
    ax4.set_xlabel('年份')
    ax4.set_ylabel('占比 (%)')
    ax4.legend(loc='center right', bbox_to_anchor=(1.15, 0.5))
    ax4.set_ylim(0, 100)
    
    plt.savefig('figures/arima_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """生成所有图表"""
    print("正在生成ARIMA预测可视化图表...")
    
    create_sectoral_chart()
    print("✓ 分部门分析图表已生成")
    
    create_intensity_chart()
    print("✓ 强度指标图表已生成")
    
    create_milestone_chart()
    print("✓ 里程碑分析图表已生成")
    
    create_policy_scenarios()
    print("✓ 政策情景图表已生成")
    
    create_dashboard()
    print("✓ 综合仪表板已生成")
    
    print("\n🎉 所有ARIMA预测图表生成完成！")
    print("\n生成的图表文件：")
    print("- figures/arima_main_forecast.png (主要预测图表)")
    print("- figures/arima_sectoral_analysis.png (分部门分析)")
    print("- figures/arima_intensity_indicators.png (强度指标分析)")
    print("- figures/arima_milestone_analysis.png (里程碑分析)")
    print("- figures/arima_policy_scenarios.png (政策情景分析)")
    print("- figures/arima_comprehensive_dashboard.png (综合仪表板)")

if __name__ == "__main__":
    main()