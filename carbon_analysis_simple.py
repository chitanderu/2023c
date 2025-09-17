#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
区域碳排放指标体系分析 - 简化版
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_and_clean_data():
    """加载和清理数据"""
    # 加载数据
    carbon_df = pd.read_csv('碳排放.csv', encoding='utf-8-sig')
    economic_df = pd.read_csv('经济与能源.csv', encoding='utf-8-sig')
    
    return carbon_df, economic_df

def build_indicator_framework():
    """建立指标体系框架"""
    indicators = {
        "一级指标": {
            "经济发展指标": {
                "GDP总量": "反映区域经济总体规模和发展水平",
                "产业结构": "三次产业增加值占GDP比重",
                "经济增长率": "GDP年度增长率",
                "人均GDP": "人均经济发展水平"
            },
            "人口社会指标": {
                "常驻人口总量": "区域人口规模",
                "人口增长率": "人口变化趋势",
                "城镇化水平": "城镇人口比重"
            },
            "能源消费指标": {
                "能源消费总量": "各类能源消费总量",
                "能源强度": "单位GDP能源消费量",
                "能源结构": "煤炭、油品、天然气等占比",
                "人均能耗": "人均能源消费量",
                "清洁能源比重": "可再生能源占比"
            },
            "碳排放指标": {
                "碳排放总量": "温室气体排放总量",
                "碳强度": "单位GDP碳排放量",
                "人均碳排放": "人均温室气体排放",
                "碳排放增长率": "碳排放年度变化率"
            }
        },
        
        "二级指标(分部门)": {
            "能源供应部门": ["发电", "供热", "其他转换"],
            "工业消费部门": ["制造业", "采矿业等"],
            "建筑消费部门": ["商业建筑", "公共建筑"],
            "交通消费部门": ["道路运输", "其他运输"],
            "居民生活消费": ["居民用电", "居民用热等"],
            "农林消费部门": ["农业", "林业"]
        }
    }
    
    return indicators

def extract_key_data(carbon_df, economic_df):
    """提取关键数据"""
    years = list(range(2010, 2021))
    year_cols = [str(year) for year in years]
    
    # 提取主要指标数据
    try:
        # GDP数据
        gdp_row = economic_df[economic_df['项目'] == 'GDP']
        gdp_data = []
        for col in year_cols:
            if col in gdp_row.columns:
                val = gdp_row[col].iloc[0]
                try:
                    gdp_data.append(float(val.replace(',', '') if isinstance(val, str) else val))
                except:
                    gdp_data.append(np.nan)
        
        # 人口数据
        pop_row = economic_df[economic_df['项目'] == '常驻人口']
        pop_data = []
        for col in year_cols:
            if col in pop_row.columns:
                val = pop_row[col].iloc[0]
                try:
                    pop_data.append(float(val.replace(',', '') if isinstance(val, str) else val))
                except:
                    pop_data.append(np.nan)
        
        # 能源消费数据
        energy_row = economic_df[economic_df['项目'] == '能源消费量']
        energy_data = []
        for col in year_cols:
            if col in energy_row.columns:
                val = energy_row[col].iloc[0]
                try:
                    energy_data.append(float(val.replace(',', '') if isinstance(val, str) else val))
                except:
                    energy_data.append(np.nan)
        
        # 碳排放数据
        carbon_row = carbon_df[carbon_df['项目'] == '碳排放量']
        carbon_data = []
        for col in year_cols:
            if col in carbon_row.columns:
                val = carbon_row[col].iloc[0]
                try:
                    carbon_data.append(float(val.replace(',', '') if isinstance(val, str) else val))
                except:
                    carbon_data.append(np.nan)
        
        # 创建主数据框
        main_df = pd.DataFrame({
            '年份': years,
            'GDP(亿元)': gdp_data,
            '人口(万人)': pop_data,
            '能源消费(万tce)': energy_data,
            '碳排放(万tCO2)': carbon_data
        })
        
        # 计算衍生指标
        main_df['人均GDP(元/人)'] = (main_df['GDP(亿元)'] * 10000) / (main_df['人口(万人)'] * 10000)
        main_df['碳强度(tCO2/万元GDP)'] = main_df['碳排放(万tCO2)'] / main_df['GDP(亿元)']
        main_df['能源强度(tce/万元GDP)'] = main_df['能源消费(万tce)'] / main_df['GDP(亿元)']
        main_df['人均能耗(tce/人)'] = main_df['能源消费(万tce)'] / (main_df['人口(万人)'] * 10000)
        main_df['人均碳排放(tCO2/人)'] = main_df['碳排放(万tCO2)'] / (main_df['人口(万人)'] * 10000)
        
        # 计算增长率
        for col in ['GDP(亿元)', '碳排放(万tCO2)', '能源消费(万tce)']:
            main_df[f'{col}_增长率(%)'] = main_df[col].pct_change() * 100
        
        return main_df
        
    except Exception as e:
        print(f"数据提取错误: {e}")
        return pd.DataFrame()

def analyze_sectoral_emissions(carbon_df):
    """分析分部门碳排放"""
    years = [str(year) for year in range(2010, 2021)]
    
    # 部门映射
    sectors = {
        '农林消费部门': ['农林消费部门'],
        '工业消费部门': ['工业消费部门'],
        '建筑消费部门': ['建筑消费部门'],
        '交通消费部门': ['交通消费部门'],
        '居民生活消费': ['居民生活消费'],
        '能源供应部门': ['发电', '供热']  # 简化处理
    }
    
    sectoral_results = {}
    
    for sector_name, keywords in sectors.items():
        sector_data = []
        for year in years:
            total = 0
            for keyword in keywords:
                # 查找包含关键词的行
                mask = (carbon_df['项目'].str.contains(keyword, na=False) | 
                       carbon_df['子项'].str.contains(keyword, na=False))
                matching_rows = carbon_df[mask]
                
                if not matching_rows.empty and year in matching_rows.columns:
                    for _, row in matching_rows.iterrows():
                        try:
                            val = row[year]
                            if pd.notna(val) and val != '-':
                                total += float(str(val).replace(',', ''))
                        except:
                            continue
            sector_data.append(total)
        
        sectoral_results[sector_name] = {
            '数据': sector_data,
            '2010年': sector_data[0] if len(sector_data) > 0 else 0,
            '2020年': sector_data[-1] if len(sector_data) > 0 else 0,
            '增长率': ((sector_data[-1] / sector_data[0] - 1) * 100) if sector_data[0] > 0 and len(sector_data) > 0 else 0
        }
    
    return sectoral_results

def build_correlation_analysis(main_df):
    """构建相关性分析和预测模型"""
    # 选择完整数据
    df_clean = main_df.dropna()
    
    if df_clean.empty:
        return {}, {}
    
    # 相关性分析
    corr_vars = ['GDP(亿元)', '人口(万人)', '能源消费(万tce)', '碳排放(万tCO2)', 
                '碳强度(tCO2/万元GDP)', '能源强度(tce/万元GDP)']
    available_vars = [var for var in corr_vars if var in df_clean.columns]
    
    correlation_matrix = df_clean[available_vars].corr()
    
    # 预测模型
    models = {}
    if len(df_clean) > 3:  # 确保有足够数据建模
        try:
            # 碳排放预测模型
            X = df_clean[['GDP(亿元)', '人口(万人)', '能源消费(万tce)']]
            y = df_clean['碳排放(万tCO2)']
            
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)
            
            models['碳排放预测'] = {
                '模型': model,
                'R²': r2,
                '预测值': y_pred,
                '实际值': y.values
            }
        except Exception as e:
            print(f"模型构建错误: {e}")
    
    return correlation_matrix, models

def generate_carbon_peak_analysis(main_df):
    """碳达峰分析"""
    if '碳排放(万tCO2)' not in main_df.columns:
        return {}
    
    carbon_series = main_df['碳排放(万tCO2)'].dropna()
    
    if len(carbon_series) < 2:
        return {}
    
    analysis = {
        '起始年份碳排放': carbon_series.iloc[0],
        '结束年份碳排放': carbon_series.iloc[-1],
        '总体变化': ((carbon_series.iloc[-1] / carbon_series.iloc[0]) - 1) * 100,
        '峰值': carbon_series.max(),
        '峰值年份': main_df.loc[carbon_series.idxmax(), '年份'],
        '是否已达峰': carbon_series.iloc[-1] < carbon_series.max(),
        '近五年平均增长率': carbon_series.tail(5).pct_change().mean() * 100
    }
    
    # 主要挑战分析
    challenges = {
        '经济增长压力': 'GDP持续增长带来的碳排放压力',
        '产业结构调整': '高耗能产业占比较高',
        '能源结构优化': '煤炭等化石能源依赖度高',
        '技术进步需求': '节能减排技术有待提升',
        '政策协调': '需要跨部门协调减排政策'
    }
    
    return analysis, challenges

def create_visualizations(main_df, sectoral_results, correlation_matrix):
    """创建可视化图表"""
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('区域碳排放指标体系分析', fontsize=16, fontweight='bold')
    
    # 1. 主要指标趋势
    if not main_df.empty:
        axes[0,0].plot(main_df['年份'], main_df['碳排放(万tCO2)'], 'o-', label='碳排放', linewidth=2)
        axes[0,0].set_title('碳排放总量趋势')
        axes[0,0].set_xlabel('年份')
        axes[0,0].set_ylabel('万tCO2')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
    
    # 2. 碳强度变化
    if '碳强度(tCO2/万元GDP)' in main_df.columns:
        axes[0,1].plot(main_df['年份'], main_df['碳强度(tCO2/万元GDP)'], 's-', color='red', linewidth=2)
        axes[0,1].set_title('碳排放强度变化')
        axes[0,1].set_xlabel('年份')
        axes[0,1].set_ylabel('tCO2/万元GDP')
        axes[0,1].grid(True, alpha=0.3)
    
    # 3. GDP与碳排放关系
    if not main_df.empty:
        scatter = axes[0,2].scatter(main_df['GDP(亿元)'], main_df['碳排放(万tCO2)'], 
                                  c=main_df['年份'], cmap='viridis', s=100, alpha=0.7)
        axes[0,2].set_title('GDP与碳排放关系')
        axes[0,2].set_xlabel('GDP(亿元)')
        axes[0,2].set_ylabel('碳排放(万tCO2)')
        plt.colorbar(scatter, ax=axes[0,2], label='年份')
    
    # 4. 相关性热力图
    if not correlation_matrix.empty:
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
        axes[1,0].set_title('指标相关性分析')
    
    # 5. 部门排放占比（2020年）
    if sectoral_results:
        sectors = list(sectoral_results.keys())
        values_2020 = [sectoral_results[s]['2020年'] for s in sectors]
        
        # 过滤掉零值
        non_zero_data = [(s, v) for s, v in zip(sectors, values_2020) if v > 0]
        if non_zero_data:
            sectors_nz, values_nz = zip(*non_zero_data)
            axes[1,1].pie(values_nz, labels=sectors_nz, autopct='%1.1f%%', startangle=90)
            axes[1,1].set_title('2020年各部门碳排放占比')
    
    # 6. 人均指标对比
    if not main_df.empty and '人均GDP(元/人)' in main_df.columns:
        ax2 = axes[1,2].twinx()
        line1 = axes[1,2].plot(main_df['年份'], main_df['人均GDP(元/人)']/10000, 'b-o', label='人均GDP(万元)')
        line2 = ax2.plot(main_df['年份'], main_df['人均碳排放(tCO2/人)'], 'r-s', label='人均碳排放')
        
        axes[1,2].set_xlabel('年份')
        axes[1,2].set_ylabel('人均GDP(万元)', color='b')
        ax2.set_ylabel('人均碳排放(tCO2)', color='r')
        axes[1,2].set_title('人均GDP与人均碳排放')
        
        # 合并图例
        lines1, labels1 = axes[1,2].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1,2].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('figures/carbon_indicator_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_comprehensive_report(indicators, main_df, sectoral_results, 
                                correlation_matrix, models, peak_analysis):
    """生成综合分析报告"""
    
    report = """
区域碳排放指标体系分析报告
================================

一、指标体系构建
{line}

本研究构建了多层次的碳排放指标体系：

1.1 一级指标体系
- 经济发展指标：包括GDP总量、产业结构、经济增长率等
- 人口社会指标：包括人口总量、增长率、城镇化水平等  
- 能源消费指标：包括能源总量、强度、结构、人均能耗等
- 碳排放指标：包括排放总量、强度、人均排放、增长率等

1.2 二级指标体系（分部门）
- 能源供应部门：发电、供热等
- 工业消费部门：制造业、采矿业等
- 建筑消费部门：商业建筑、公共建筑等
- 交通消费部门：道路运输等
- 居民生活消费：生活用能等
- 农林消费部门：农业、林业等

二、区域现状分析（2010-2020年）
{line}
""".format(line='-'*50)
    
    if not main_df.empty:
        report += f"""
2.1 主要指标变化
- 碳排放总量：{main_df['碳排放(万tCO2)'].iloc[0]:.1f} → {main_df['碳排放(万tCO2)'].iloc[-1]:.1f} 万tCO2
- GDP总量：{main_df['GDP(亿元)'].iloc[0]:.1f} → {main_df['GDP(亿元)'].iloc[-1]:.1f} 亿元  
- 人口规模：{main_df['人口(万人)'].iloc[0]:.1f} → {main_df['人口(万人)'].iloc[-1]:.1f} 万人
- 碳排放强度：{main_df['碳强度(tCO2/万元GDP)'].iloc[0]:.3f} → {main_df['碳强度(tCO2/万元GDP)'].iloc[-1]:.3f} tCO2/万元GDP
- 人均碳排放：{main_df['人均碳排放(tCO2/人)'].iloc[0]:.2f} → {main_df['人均碳排放(tCO2/人)'].iloc[-1]:.2f} tCO2/人

2.2 发展阶段特征
- 十二五期间(2011-2015)：快速增长阶段
- 十三五期间(2016-2020)：增速放缓阶段
"""
    
    if sectoral_results:
        report += f"""
三、分部门碳排放分析
{'-'*50}

各部门2020年排放状况及十年变化：
"""
        for sector, data in sectoral_results.items():
            if data['2020年'] > 0:
                report += f"""
{sector}：
- 2020年排放量：{data['2020年']:.1f} 万tCO2
- 十年增长率：{data['增长率']:.1f}%
"""
    
    if peak_analysis:
        report += f"""
四、碳达峰分析
{'-'*50}

4.1 达峰情况判断
- 是否已达峰：{peak_analysis[0]['是否已达峰']}
- 峰值排放量：{peak_analysis[0]['峰值']:.1f} 万tCO2
- 峰值年份：{peak_analysis[0]['峰值年份']}
- 总体增长：{peak_analysis[0]['总体变化']:.1f}%

4.2 主要挑战
"""
        for challenge, description in peak_analysis[1].items():
            report += f"- {challenge}：{description}\n"
    
    if models and '碳排放预测' in models:
        report += f"""
五、预测模型评估
{'-'*50}

碳排放预测模型：
- 模型类型：多元线性回归
- 解释变量：GDP、人口、能源消费
- 模型R²：{models['碳排放预测']['R²']:.3f}
- 模型适用性：{'良好' if models['碳排放预测']['R²'] > 0.8 else '一般' if models['碳排放预测']['R²'] > 0.6 else '需改进'}
"""
    
    report += f"""
六、政策建议
{'-'*50}

6.1 短期措施（1-3年）
- 优化能源结构，提高清洁能源比重
- 推进重点行业节能减排技术改造
- 完善碳排放统计监测体系

6.2 中期目标（3-5年）
- 加快产业结构转型升级
- 建立区域碳交易机制
- 推动绿色建筑和绿色交通发展

6.3 长期规划（5-10年）
- 实现能源体系深度脱碳
- 建立完善的碳中和技术体系
- 构建区域协调的双碳实现路径

七、结论
{'-'*50}

通过建立综合指标体系分析发现，该区域在碳减排方面既面临挑战也具备潜力。
需要统筹经济发展与碳减排目标，制定差异化的减排路径，确保如期实现
碳达峰和碳中和目标。

报告生成时间：2024年
""".format(line='-'*50)
    
    # 保存报告
    with open('outputs/comprehensive_carbon_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report

def main():
    """主分析流程"""
    print("开始区域碳排放指标体系分析...")
    
    # 创建输出目录
    import os
    for dirname in ['outputs', 'figures']:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    
    # 1. 加载数据
    carbon_df, economic_df = load_and_clean_data()
    
    # 2. 构建指标体系
    indicators = build_indicator_framework()
    
    # 3. 提取关键数据
    main_df = extract_key_data(carbon_df, economic_df)
    
    # 4. 分部门排放分析
    sectoral_results = analyze_sectoral_emissions(carbon_df)
    
    # 5. 相关性分析和建模
    correlation_matrix, models = build_correlation_analysis(main_df)
    
    # 6. 碳达峰分析
    peak_analysis = generate_carbon_peak_analysis(main_df)
    
    # 7. 创建可视化
    create_visualizations(main_df, sectoral_results, correlation_matrix)
    
    # 8. 生成报告
    report = generate_comprehensive_report(indicators, main_df, sectoral_results,
                                         correlation_matrix, models, peak_analysis)
    
    print("分析完成！")
    print("生成的文件：")
    print("- outputs/comprehensive_carbon_report.txt (分析报告)")
    print("- figures/carbon_indicator_analysis.png (可视化图表)")
    
    return {
        'indicators': indicators,
        'main_data': main_df,
        'sectoral_results': sectoral_results,
        'correlation_matrix': correlation_matrix,
        'models': models,
        'peak_analysis': peak_analysis
    }

if __name__ == "__main__":
    results = main()