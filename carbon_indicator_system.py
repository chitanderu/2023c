#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
区域碳排放指标体系分析
基于2010-2020年经济、人口、能源消费和碳排放数据
"""W

import pandas as pd
import numpy as npW
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preproceWssing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class CarbonIndicatorSystem:
    def __init__(self):
        self.carbon_data = None
        self.economic_data = None
        self.indicator_system = {}
        
    def load_data(self):
        """加载碳排放和经济能源数据"""
        try:
            self.carbon_data = pd.read_csv('碳排放.csv', encoding='utf-8-sig')
            self.economic_data = pd.read_csv('经济与能源.csv', encoding='utf-8-sig')
            print("数据加载成功")
        except Exception as e:
            print(f"数据加载失败: {e}")
    
    def build_indicator_system(self):
        """构建四大类指标体系"""
        
        # 1. 经济发展指标
        economic_indicators = {
            'GDP总量': '反映区域经济总体规模',
            'GDP增长率': '反映经济发展速度',
            '产业结构': '三次产业比重',
            '人均GDP': '经济发展水平',
            '工业增加值': '工业发展规模',
            '能源供应部门产值': '能源产业贡献'
        }
        
        # 2. 人口社会指标
        population_indicators = {
            '常驻人口总量': '人口规模',
            '人口增长率': '人口变化趋势',
            '人口密度': '人口分布密度',
            '城镇化率': '城镇发展水平'
        }
        
        # 3. 能源消费指标
        energy_indicators = {
            '能源消费总量': '能源需求规模',
            '能源消费结构': '各类能源比重',
            '能源强度': '单位GDP能耗',
            '人均能耗': '人均能源消费',
            '清洁能源比重': '能源清洁化程度',
            '外调电力依赖度': '能源供应安全'
        }
        
        # 4. 碳排放指标
        carbon_indicators = {
            '碳排放总量': '温室气体排放规模',
            '碳排放强度': '单位GDP碳排放',
            '人均碳排放': '人均温室气体排放',
            '分部门碳排放': '各部门排放贡献',
            '碳排放增长率': '排放变化趋势',
            '碳排放因子': '排放效率指标'
        }
        
        self.indicator_system = {
            '经济发展指标': economic_indicators,
            '人口社会指标': population_indicators,
            '能源消费指标': energy_indicators,
            '碳排放指标': carbon_indicators
        }
        
        return self.indicator_system
    
    def calculate_derived_indicators(self):
        """计算衍生指标"""
        years = [str(year) for year in range(2010, 2021)]
        
        # 提取基础数据
        gdp_data = self.economic_data[self.economic_data['项目'] == 'GDP'].iloc[:, 5:16].values.flatten()
        population_data = self.economic_data[self.economic_data['项目'] == '常驻人口'].iloc[:, 5:16].values.flatten()
        energy_total = self.economic_data[self.economic_data['项目'] == '能源消费量'].iloc[:, 5:16].values.flatten()
        carbon_total = self.carbon_data[self.carbon_data['项目'] == '碳排放量'].iloc[:, 5:16].values.flatten()
        
        # 计算衍生指标
        derived_indicators = pd.DataFrame({
            '年份': years,
            'GDP(亿元)': gdp_data,
            '人口(万人)': population_data,
            '能源消费(万tce)': energy_total,
            '碳排放(万tCO2)': carbon_total,
            '人均GDP(元/人)': (gdp_data * 10000) / (population_data * 10000),
            '人均能耗(tce/人)': energy_total / (population_data * 10000),
            '人均碳排放(tCO2/人)': carbon_total / (population_data * 10000),
            '碳强度(tCO2/万元GDP)': carbon_total / gdp_data,
            '能源强度(tce/万元GDP)': energy_total / gdp_data,
            '碳排放系数(tCO2/tce)': carbon_total / energy_total
        })
        
        # 计算同比增长率
        for col in ['GDP(亿元)', '人口(万人)', '能源消费(万tce)', '碳排放(万tCO2)']:
            derived_indicators[f'{col}_增长率'] = derived_indicators[col].pct_change() * 100
        
        self.derived_indicators = derived_indicators
        return derived_indicators
    
    def analyze_sectoral_emissions(self):
        """分析六大部门碳排放状况"""
        sectors = {
            '能源供应部门': ['发电', '供热', '其他转换'],
            '工业消费部门': ['工业消费部门'],
            '建筑消费部门': ['建筑消费部门'],
            '交通消费部门': ['交通消费部门'],
            '居民生活消费': ['居民生活消费'],
            '农林消费部门': ['农林消费部门']
        }
        
        years = [str(year) for year in range(2010, 2021)]
        sectoral_data = {}
        
        for sector, keywords in sectors.items():
            sector_emissions = []
            for keyword in keywords:
                mask = (self.carbon_data['项目'].str.contains(keyword, na=False) | 
                       self.carbon_data['子项'].str.contains(keyword, na=False))
                if mask.any():
                    data = self.carbon_data[mask].iloc[:, 5:16].values
                    if len(data) > 0:
                        sector_emissions.append(data[0])
            
            if sector_emissions:
                sectoral_data[sector] = np.sum(sector_emissions, axis=0)
            else:
                sectoral_data[sector] = np.zeros(11)
        
        sectoral_df = pd.DataFrame(sectoral_data, index=years)
        
        # 计算各部门占比和增长趋势
        sectoral_analysis = {}
        for sector in sectoral_data.keys():
            data = sectoral_df[sector]
            sectoral_analysis[sector] = {
                '2010年排放量': data.iloc[0],
                '2020年排放量': data.iloc[-1],
                '十年增长率': ((data.iloc[-1] / data.iloc[0]) - 1) * 100 if data.iloc[0] > 0 else 0,
                '2020年占比': (data.iloc[-1] / sectoral_df.sum(axis=1).iloc[-1]) * 100,
                '平均年增长率': data.pct_change().mean() * 100
            }
        
        self.sectoral_analysis = sectoral_analysis
        self.sectoral_df = sectoral_df
        return sectoral_analysis, sectoral_df
    
    def build_correlation_model(self):
        """建立指标间关联关系模型"""
        df = self.derived_indicators.copy()
        
        # 选择关键变量进行相关性分析
        key_vars = ['GDP(亿元)', '人口(万人)', '能源消费(万tce)', '碳排放(万tCO2)',
                   '人均GDP(元/人)', '碳强度(tCO2/万元GDP)', '能源强度(tce/万元GDP)']
        
        correlation_matrix = df[key_vars].corr()
        
        # 建立碳排放预测模型
        X = df[['GDP(亿元)', '人口(万人)', '能源消费(万tce)']].dropna()
        y = df['碳排放(万tCO2)'].dropna()
        
        # 线性回归模型
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        lr_pred = lr_model.predict(X)
        lr_r2 = r2_score(y, lr_pred)
        
        # 随机森林模型
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        rf_pred = rf_model.predict(X)
        rf_r2 = r2_score(y, rf_pred)
        
        self.correlation_matrix = correlation_matrix
        self.models = {
            'linear_regression': {'model': lr_model, 'r2': lr_r2, 'predictions': lr_pred},
            'random_forest': {'model': rf_model, 'r2': rf_r2, 'predictions': rf_pred}
        }
        
        return correlation_matrix, self.models
    
    def carbon_peak_analysis(self):
        """碳达峰分析"""
        carbon_series = self.derived_indicators['碳排放(万tCO2)']
        carbon_growth = self.derived_indicators['碳排放(万tCO2)_增长率'].dropna()
        
        analysis_results = {
            '2010-2015年平均增长率': carbon_growth.iloc[1:6].mean(),
            '2015-2020年平均增长率': carbon_growth.iloc[6:].mean(),
            '2020年相比2010年增长': ((carbon_series.iloc[-1] / carbon_series.iloc[0]) - 1) * 100,
            '峰值年份': carbon_series.idxmax(),
            '峰值排放量': carbon_series.max(),
            '是否已达峰': carbon_series.iloc[-1] < carbon_series.max()
        }
        
        # 主要挑战识别
        challenges = {
            '经济增长与碳排放脱钩': self.derived_indicators['碳强度(tCO2/万元GDP)'].iloc[-1] > 
                                  self.derived_indicators['碳强度(tCO2/万元GDP)'].iloc[0],
            '能源结构优化': '需要提高清洁能源比重',
            '产业结构调整': '需要降低高耗能产业比重',
            '技术进步': '需要提升能源利用效率',
            '政策执行': '需要强化碳减排政策'
        }
        
        self.peak_analysis = analysis_results
        self.challenges = challenges
        
        return analysis_results, challenges
    
    def generate_visualization(self):
        """生成可视化图表"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. 碳排放总量趋势
        years = list(range(2010, 2021))
        carbon_total = self.derived_indicators['碳排放(万tCO2)']
        axes[0,0].plot(years, carbon_total, marker='o', linewidth=2, markersize=6)
        axes[0,0].set_title('碳排放总量变化趋势', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('年份')
        axes[0,0].set_ylabel('碳排放量(万tCO2)')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 分部门碳排放占比
        if hasattr(self, 'sectoral_df'):
            sector_2020 = self.sectoral_df.iloc[-1]
            axes[0,1].pie(sector_2020.values, labels=sector_2020.index, autopct='%1.1f%%')
            axes[0,1].set_title('2020年各部门碳排放占比', fontsize=14, fontweight='bold')
        
        # 3. 碳强度变化
        carbon_intensity = self.derived_indicators['碳强度(tCO2/万元GDP)']
        axes[0,2].plot(years, carbon_intensity, marker='s', color='red', linewidth=2)
        axes[0,2].set_title('碳排放强度变化', fontsize=14, fontweight='bold')
        axes[0,2].set_xlabel('年份')
        axes[0,2].set_ylabel('碳强度(tCO2/万元GDP)')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. 相关性热力图
        if hasattr(self, 'correlation_matrix'):
            sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
            axes[1,0].set_title('指标相关性分析', fontsize=14, fontweight='bold')
        
        # 5. GDP与碳排放关系
        gdp = self.derived_indicators['GDP(亿元)']
        carbon = self.derived_indicators['碳排放(万tCO2)']
        axes[1,1].scatter(gdp, carbon, s=100, alpha=0.7, c=years, cmap='viridis')
        axes[1,1].set_xlabel('GDP(亿元)')
        axes[1,1].set_ylabel('碳排放(万tCO2)')
        axes[1,1].set_title('GDP与碳排放关系', fontsize=14, fontweight='bold')
        
        # 6. 人均指标对比
        per_capita_gdp = self.derived_indicators['人均GDP(元/人)'] / 10000
        per_capita_carbon = self.derived_indicators['人均碳排放(tCO2/人)']
        
        ax2 = axes[1,2].twinx()
        line1 = axes[1,2].plot(years, per_capita_gdp, 'b-', marker='o', label='人均GDP(万元)')
        line2 = ax2.plot(years, per_capita_carbon, 'r-', marker='s', label='人均碳排放(tCO2)')
        
        axes[1,2].set_xlabel('年份')
        axes[1,2].set_ylabel('人均GDP(万元)', color='b')
        ax2.set_ylabel('人均碳排放(tCO2)', color='r')
        axes[1,2].set_title('人均GDP与人均碳排放', fontsize=14, fontweight='bold')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[1,2].legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        plt.savefig('figures/carbon_analysis_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """生成分析报告"""
        report = f"""
区域碳排放指标体系分析报告
================================

一、指标体系构建
{'-'*50}
本研究构建了四大类指标体系：

1. 经济发展指标（7项）
   - GDP总量、GDP增长率、产业结构等

2. 人口社会指标（4项）
   - 常驻人口总量、人口增长率等

3. 能源消费指标（6项）
   - 能源消费总量、能源消费结构等

4. 碳排放指标（6项）
   - 碳排放总量、碳排放强度等

二、现状分析（2010-2020年）
{'-'*50}
"""
        
        if hasattr(self, 'derived_indicators'):
            df = self.derived_indicators
            report += f"""
关键指标变化：
- 碳排放总量：{df['碳排放(万tCO2)'].iloc[0]:.1f} → {df['碳排放(万tCO2)'].iloc[-1]:.1f} 万tCO2
- GDP总量：{df['GDP(亿元)'].iloc[0]:.1f} → {df['GDP(亿元)'].iloc[-1]:.1f} 亿元
- 碳强度：{df['碳强度(tCO2/万元GDP)'].iloc[0]:.3f} → {df['碳强度(tCO2/万元GDP)'].iloc[-1]:.3f} tCO2/万元
- 人均碳排放：{df['人均碳排放(tCO2/人)'].iloc[0]:.2f} → {df['人均碳排放(tCO2/人)'].iloc[-1]:.2f} tCO2/人
"""
        
        if hasattr(self, 'peak_analysis'):
            report += f"""
三、碳达峰分析
{'-'*50}
- 2010-2015年平均增长率：{self.peak_analysis['2010-2015年平均增长率']:.2f}%
- 2015-2020年平均增长率：{self.peak_analysis['2015-2020年平均增长率']:.2f}%
- 是否已达峰：{self.peak_analysis['是否已达峰']}
- 峰值排放量：{self.peak_analysis['峰值排放量']:.1f} 万tCO2

主要挑战：
"""
            for challenge, description in self.challenges.items():
                report += f"- {challenge}：{description}\n"
        
        if hasattr(self, 'sectoral_analysis'):
            report += f"""

四、分部门排放分析
{'-'*50}
"""
            for sector, data in self.sectoral_analysis.items():
                report += f"""
{sector}：
- 2020年排放量：{data['2020年排放量']:.1f} 万tCO2
- 2020年占比：{data['2020年占比']:.1f}%
- 十年增长率：{data['十年增长率']:.1f}%
"""
        
        if hasattr(self, 'models'):
            report += f"""

五、预测模型性能
{'-'*50}
- 线性回归模型 R²：{self.models['linear_regression']['r2']:.3f}
- 随机森林模型 R²：{self.models['random_forest']['r2']:.3f}
"""
        
        report += """

六、政策建议
{'-'*50}
1. 加快产业结构调整，发展低碳产业
2. 优化能源结构，提高清洁能源比重
3. 提升能源利用效率，推进节能技术
4. 建立完善的碳排放监测体系
5. 制定差异化的碳减排路径
"""
        
        # 保存报告
        with open('outputs/carbon_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report

def main():
    """主函数"""
    # 创建输出目录
    import os
    for dir_name in ['outputs', 'figures']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    
    # 初始化分析系统
    analyzer = CarbonIndicatorSystem()
    
    # 加载数据
    analyzer.load_data()
    
    # 构建指标体系
    indicator_system = analyzer.build_indicator_system()
    
    # 计算衍生指标
    derived_indicators = analyzer.calculate_derived_indicators()
    
    # 分部门排放分析
    sectoral_analysis, sectoral_df = analyzer.analyze_sectoral_emissions()
    
    # 建立关联模型
    correlation_matrix, models = analyzer.build_correlation_model()
    
    # 碳达峰分析
    peak_analysis, challenges = analyzer.carbon_peak_analysis()
    
    # 生成可视化
    analyzer.generate_visualization()
    
    # 生成报告
    report = analyzer.generate_report()
    
    print("分析完成！")
    print("生成文件：")
    print("- outputs/carbon_analysis_report.txt")
    print("- figures/carbon_analysis_overview.png")
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()