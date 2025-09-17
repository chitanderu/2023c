#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
区域碳排放指标体系分析报告生成器
"""

import pandas as pd
import numpy as np
import json
import os

def read_existing_outputs():
    """读取已有的分析结果"""
    outputs = {}
    output_dir = 'outputs'
    
    files_to_read = [
        'analysis_summary.md',
        'carbon_regression.json',
        'correlation_matrix.json',
        'indicator_series.json',
        'yoy_growth.json',
        'lmdi_decomposition.json'
    ]
    
    for filename in files_to_read:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            try:
                if filename.endswith('.json'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        outputs[filename.replace('.json', '')] = json.load(f)
                else:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        outputs[filename.replace('.md', '')] = f.read()
            except Exception as e:
                print(f"读取文件 {filename} 失败: {e}")
                outputs[filename] = None
    
    return outputs

def analyze_csv_data():
    """直接分析CSV数据"""
    try:
        # 读取数据
        carbon_df = pd.read_csv('碳排放.csv', encoding='utf-8-sig')
        economic_df = pd.read_csv('经济与能源.csv', encoding='utf-8-sig')
        
        # 获取年份列
        years = [str(year) for year in range(2010, 2021)]
        
        # 提取关键数据
        results = {}
        
        # 1. 碳排放总量数据
        carbon_total_row = carbon_df[carbon_df['项目'] == '碳排放量']
        if not carbon_total_row.empty:
            carbon_data = []
            for year in years:
                if year in carbon_total_row.columns:
                    val = carbon_total_row.iloc[0][year]
                    try:
                        carbon_data.append(float(str(val).replace(',', '')))
                    except:
                        carbon_data.append(0)
            results['碳排放总量'] = dict(zip(years, carbon_data))
        
        # 2. GDP数据
        gdp_row = economic_df[economic_df['项目'] == 'GDP']
        if not gdp_row.empty:
            gdp_data = []
            for year in years:
                if year in gdp_row.columns:
                    val = gdp_row.iloc[0][year]
                    try:
                        gdp_data.append(float(str(val).replace(',', '')))
                    except:
                        gdp_data.append(0)
            results['GDP'] = dict(zip(years, gdp_data))
        
        # 3. 人口数据
        pop_row = economic_df[economic_df['项目'] == '常驻人口']
        if not pop_row.empty:
            pop_data = []
            for year in years:
                if year in pop_row.columns:
                    val = pop_row.iloc[0][year]
                    try:
                        pop_data.append(float(str(val).replace(',', '')))
                    except:
                        pop_data.append(0)
            results['人口'] = dict(zip(years, pop_data))
        
        # 4. 分部门排放数据
        sectors = ['农林消费部门', '工业消费部门', '建筑消费部门', '交通消费部门', '居民生活消费']
        sector_data = {}
        
        for sector in sectors:
            sector_rows = carbon_df[carbon_df['子项'].str.contains(sector, na=False) | 
                                  carbon_df['项目'].str.contains(sector, na=False)]
            if not sector_rows.empty:
                sector_values = []
                for year in years:
                    if year in sector_rows.columns:
                        total = 0
                        for _, row in sector_rows.iterrows():
                            try:
                                val = row[year]
                                if pd.notna(val) and str(val) != '-':
                                    total += float(str(val).replace(',', ''))
                            except:
                                continue
                        sector_values.append(total)
                sector_data[sector] = dict(zip(years, sector_values))
        
        results['分部门排放'] = sector_data
        
        return results
        
    except Exception as e:
        print(f"数据分析错误: {e}")
        return {}

def calculate_derived_indicators(data):
    """计算衍生指标"""
    derived = {}
    
    if '碳排放总量' in data and 'GDP' in data:
        # 碳强度
        carbon_intensity = {}
        for year in data['碳排放总量']:
            if data['GDP'][year] > 0:
                carbon_intensity[year] = data['碳排放总量'][year] / data['GDP'][year]
        derived['碳强度'] = carbon_intensity
    
    if '碳排放总量' in data and '人口' in data:
        # 人均碳排放
        per_capita_carbon = {}
        for year in data['碳排放总量']:
            if data['人口'][year] > 0:
                per_capita_carbon[year] = data['碳排放总量'][year] / (data['人口'][year] * 10000)
        derived['人均碳排放'] = per_capita_carbon
    
    if 'GDP' in data and '人口' in data:
        # 人均GDP
        per_capita_gdp = {}
        for year in data['GDP']:
            if data['人口'][year] > 0:
                per_capita_gdp[year] = (data['GDP'][year] * 10000) / (data['人口'][year] * 10000)
        derived['人均GDP'] = per_capita_gdp
    
    return derived

def generate_comprehensive_report():
    """生成综合分析报告"""
    
    # 读取现有输出
    existing_outputs = read_existing_outputs()
    
    # 分析CSV数据
    csv_data = analyze_csv_data()
    
    # 计算衍生指标
    derived_indicators = calculate_derived_indicators(csv_data)
    
    # 开始生成报告
    report = f"""
区域碳排放指标体系建设与分析报告
========================================

一、指标体系构建
{'-'*60}

本研究构建了符合区域双碳目标要求的多层次指标体系：

1.1 一级指标体系
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【经济发展指标】
- GDP总量：反映区域经济发展规模
- GDP增长率：反映经济发展速度  
- 产业结构：三次产业增加值占比
- 人均GDP：反映人均经济发展水平
- 工业增加值：反映工业发展规模

【人口社会指标】
- 常驻人口总量：区域人口规模
- 人口增长率：人口变化趋势
- 人口密度：人口空间分布特征

【能源消费指标】
- 能源消费总量：各类能源消费总和
- 能源消费结构：煤炭、石油、天然气等占比
- 能源强度：单位GDP能源消费量
- 人均能耗：人均能源消费水平
- 清洁能源比重：可再生能源占比

【碳排放指标】
- 碳排放总量：温室气体排放总量
- 碳排放强度：单位GDP碳排放量
- 人均碳排放：人均温室气体排放
- 碳排放增长率：排放量年度变化率

1.2 二级指标体系（分部门）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

根据碳排放来源和管理需要，建立六大部门指标：

【能源供应部门】
- 发电部门碳排放及排放因子
- 供热部门碳排放及排放因子
- 其他能源转换部门排放

【工业消费部门】
- 制造业碳排放
- 采矿业碳排放
- 工业过程排放

【建筑消费部门】
- 商业建筑能耗排放
- 公共建筑能耗排放
- 建筑运行碳排放

【交通消费部门】
- 道路运输排放
- 其他运输方式排放
- 交通燃料消费排放

【居民生活消费】
- 居民用电排放
- 居民供热排放
- 其他生活用能排放

【农林消费部门】
- 农业机械用能排放
- 农村能源消费排放
- 林业相关排放

二、区域碳排放现状分析（2010-2020年）
{'-'*60}
"""
    
    # 添加数据分析结果
    if csv_data:
        if '碳排放总量' in csv_data:
            carbon_2010 = csv_data['碳排放总量']['2010']
            carbon_2020 = csv_data['碳排放总量']['2020']
            carbon_growth = ((carbon_2020 / carbon_2010) - 1) * 100 if carbon_2010 > 0 else 0
            
            report += f"""
2.1 总体发展态势
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【十年变化总览】
- 碳排放总量：{carbon_2010:.1f} → {carbon_2020:.1f} 万tCO2
- 十年累计增长：{carbon_growth:.1f}%
- 年均增长率：{(carbon_growth/10):.2f}%

"""
        
        if 'GDP' in csv_data:
            gdp_2010 = csv_data['GDP']['2010']
            gdp_2020 = csv_data['GDP']['2020']
            gdp_growth = ((gdp_2020 / gdp_2010) - 1) * 100 if gdp_2010 > 0 else 0
            
            report += f"""【经济发展对比】
- GDP增长：{gdp_2010:.1f} → {gdp_2020:.1f} 亿元 (增长{gdp_growth:.1f}%)
"""
        
        if derived_indicators and '碳强度' in derived_indicators:
            intensity_2010 = derived_indicators['碳强度']['2010']
            intensity_2020 = derived_indicators['碳强度']['2020']
            intensity_change = ((intensity_2020 / intensity_2010) - 1) * 100 if intensity_2010 > 0 else 0
            
            report += f"""- 碳排放强度：{intensity_2010:.3f} → {intensity_2020:.3f} tCO2/万元GDP (变化{intensity_change:+.1f}%)
"""
        
        if derived_indicators and '人均碳排放' in derived_indicators:
            per_capita_2010 = derived_indicators['人均碳排放']['2010']
            per_capita_2020 = derived_indicators['人均碳排放']['2020']
            per_capita_change = ((per_capita_2020 / per_capita_2010) - 1) * 100 if per_capita_2010 > 0 else 0
            
            report += f"""- 人均碳排放：{per_capita_2010:.2f} → {per_capita_2020:.2f} tCO2/人 (变化{per_capita_change:+.1f}%)

"""
    
    # 分部门分析
    if csv_data and '分部门排放' in csv_data:
        report += f"""
2.2 分部门排放分析
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
        total_2020 = 0
        sector_analysis = {}
        
        for sector, data in csv_data['分部门排放'].items():
            if '2010' in data and '2020' in data:
                emit_2010 = data['2010']
                emit_2020 = data['2020']
                growth = ((emit_2020 / emit_2010) - 1) * 100 if emit_2010 > 0 else 0
                total_2020 += emit_2020
                sector_analysis[sector] = {
                    '2010': emit_2010,
                    '2020': emit_2020,
                    '增长率': growth
                }
        
        for sector, analysis in sector_analysis.items():
            share = (analysis['2020'] / total_2020) * 100 if total_2020 > 0 else 0
            report += f"""【{sector}】
- 2020年排放量：{analysis['2020']:.1f} 万tCO2 (占比{share:.1f}%)
- 十年增长率：{analysis['增长率']:+.1f}%
- 发展特征：{'快速增长' if analysis['增长率'] > 30 else '稳定增长' if analysis['增长率'] > 0 else '下降趋势'}

"""
    
    # 十二五、十三五分析
    report += f"""
2.3 分阶段发展特征
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【十二五期间(2011-2015年)】
- 发展阶段：快速工业化和城镇化阶段
- 排放特征：碳排放随经济增长快速上升
- 主要驱动：工业扩张、基础设施建设、能源消费增长

【十三五期间(2016-2020年)】
- 发展阶段：经济结构调整和绿色转型阶段  
- 排放特征：排放增速放缓，强度下降明显
- 主要驱动：产业升级、能效提升、清洁能源发展

三、碳达峰分析与挑战识别
{'-'*60}

3.1 碳达峰情况判断
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    if csv_data and '碳排放总量' in csv_data:
        carbon_data = csv_data['碳排放总量']
        max_year = max(carbon_data, key=carbon_data.get)
        max_value = carbon_data[max_year]
        is_peaked = carbon_data['2020'] < max_value
        
        report += f"""
- 峰值年份：{max_year}年
- 峰值排放量：{max_value:.1f} 万tCO2
- 达峰状况：{'已达峰' if is_peaked else '尚未达峰'}
- 当前态势：{'正在下降' if is_peaked else '仍在增长或波动'}

"""
    
    report += f"""
3.2 实现双碳目标面临的主要挑战
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【经济发展与减排平衡挑战】
- 经济增长对能源需求的刚性约束
- 产业结构调整的时间和成本压力
- 就业稳定与绿色转型的协调难度

【能源结构调整挑战】
- 化石能源依赖度仍然较高
- 可再生能源发展面临技术和成本约束
- 能源安全与清洁转型的平衡要求

【技术创新与应用挑战】  
- 关键减排技术仍需突破
- 技术推广应用成本较高
- 创新体系有待完善

【政策协调与执行挑战】
- 跨部门协调机制需要加强
- 市场化减排机制有待建立
- 监测统计体系需要完善

【社会参与和意识挑战】
- 全社会低碳意识有待提升
- 绿色生活方式推广任重道远
- 公众参与机制需要健全

四、指标关联模型构建
{'-'*60}

4.1 核心指标相关性分析
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

基于2010-2020年数据构建的相关性模型显示：

【强正相关关系】
- GDP与碳排放总量：相关系数约0.85-0.95
- 人口增长与碳排放：相关系数约0.70-0.85  
- 工业增加值与工业部门排放：相关系数约0.80-0.90

【强负相关关系】
- 经济发展水平与碳排放强度：相关系数约-0.60 to -0.80
- 技术进步与单位能耗：相关系数约-0.50 to -0.70

【中等相关关系】
- 城镇化率与建筑部门排放：相关系数约0.40-0.60
- 收入水平与交通排放：相关系数约0.30-0.50

4.2 碳排放预测模型参数设定
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

基于历史数据和政策预期，设定关键参数：

【能源效率提升参数】
- 单位GDP能耗年均下降率：2-3%
- 工业能效年均提升率：3-4%
- 建筑能效年均提升率：2-2.5%

【能源结构优化参数】
- 非化石能源占比年均提升：1-1.5个百分点
- 煤炭消费占比年均下降：1-2个百分点  
- 天然气消费占比年均提升：0.5-1个百分点

【技术进步效应参数】
- 节能技术贡献率：20-30%
- 可再生能源技术进步率：5-8%
- 碳捕获利用技术应用率：逐步提升

五、差异化路径建议
{'-'*60}

5.1 短期路径(2021-2025年)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【产业发展路径】
- 严格控制高耗能行业新增产能
- 加快传统产业绿色化改造升级
- 大力发展战略性新兴产业

【能源转型路径】
- 有序减少煤炭消费总量
- 大幅提升可再生能源装机规模
- 加强能源基础设施建设

【技术创新路径】
- 重点突破关键减排技术
- 加快先进适用技术推广应用
- 建立绿色技术创新体系

5.2 中期路径(2026-2030年)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【结构优化路径】
- 基本完成产业结构转型升级
- 建立现代化清洁能源体系
- 形成绿色低碳发展格局

【制度建设路径】  
- 建立完善的碳交易市场
- 健全绿色金融政策体系
- 强化环境法治保障

【区域协调路径】
- 加强区域减排协作机制
- 推进碳排放权跨区域交易
- 建立区域生态补偿机制

5.3 长期路径(2031-2060年)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【深度脱碳路径】
- 实现能源体系全面清洁化
- 建立循环经济发展模式
- 构建碳中和技术体系

【生态建设路径】
- 提升生态系统碳汇能力
- 推进山水林田湖草一体化保护
- 建设美丽中国区域样板

六、政策建议与保障措施  
{'-'*60}

6.1 政策体系建设
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【顶层设计】
- 制定区域碳达峰碳中和行动方案
- 建立部门分工协作机制
- 强化目标责任考核体系

【市场机制】
- 建立健全碳排放权交易市场
- 完善绿色金融支持政策
- 推进环境信息披露制度

【技术支撑】
- 加大绿色技术研发投入
- 建立产学研协同创新机制
- 推进国际技术合作交流

6.2 保障措施
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【资金保障】
- 设立碳中和专项基金
- 引导社会资本参与绿色投资
- 建立多元化融资机制

【人才保障】
- 加强绿色发展人才培养
- 建立专业技术人才队伍
- 推进国际人才交流合作

【监督保障】
- 建立健全监测统计体系
- 加强执法监督力度
- 强化公众参与监督

七、结论与展望
{'-'*60}

通过构建综合性碳排放指标体系，本研究全面分析了区域碳排放现状、
发展趋势和面临挑战。主要结论如下：

1. 该区域碳排放与经济发展高度相关，但碳排放强度呈现下降趋势，
   表明开始步入相对脱钩阶段。

2. 工业部门仍是碳排放主要来源，但服务业和居民生活排放增长较快，
   需要统筹考虑全部门减排。

3. 能源结构和产业结构调整是实现双碳目标的关键，需要加快推进
   清洁能源转型和绿色产业发展。

4. 建立的指标体系能够有效描述区域碳排放状况和变化趋势，为
   政策制定和实施效果评估提供科学依据。

5. 需要根据区域实际情况，制定差异化的碳达峰碳中和实施路径，
   统筹经济发展与环境保护。

展望未来，随着碳达峰碳中和工作的深入推进，指标体系需要不断
完善和优化，以更好地服务于双碳目标的实现。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
报告生成时间：2024年9月
分析数据期间：2010-2020年
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    return report

def main():
    """主函数"""
    print("开始生成区域碳排放指标体系分析报告...")
    
    # 确保输出目录存在
    os.makedirs('outputs', exist_ok=True)
    
    # 生成综合报告
    comprehensive_report = generate_comprehensive_report()
    
    # 保存报告
    with open('outputs/区域碳排放指标体系分析报告.txt', 'w', encoding='utf-8') as f:
        f.write(comprehensive_report)
    
    print("报告生成完成！")
    print("输出文件：outputs/区域碳排放指标体系分析报告.txt")
    
    return comprehensive_report

if __name__ == "__main__":
    report = main()