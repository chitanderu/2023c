"""Analysis of regional carbon emissions and economic-energy indicators."""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import statistics

YEARS: List[int] = list(range(2010, 2021))


@dataclass
class DataRow:
    """Represents one logical data row extracted from the CSV."""

    theme: str
    project: str
    subitem: str
    detail: Optional[str]
    unit: str
    values: Dict[int, float]

    def series(self) -> List[float]:
        return [self.values.get(year, 0.0) for year in YEARS]


class DataTable:
    """Simple CSV loader that keeps contextual hierarchy between rows."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.rows: List[DataRow] = []
        self._load()

    def _load(self) -> None:
        last_theme = ""
        last_project = ""
        last_subitem = ""
        with open(self.path, encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                theme = (raw.get("主题") or "").strip()
                project = (raw.get("项目") or "").strip()
                subitem = (raw.get("子项") or "").strip()
                detail = (raw.get("细分项") or "").strip()
                unit = (raw.get("单位") or "").strip()

                if not any([theme, project, subitem, detail]):
                    continue
                if theme.startswith("注释") or project.startswith("注释") or subitem.startswith("注释") or detail.startswith("注释"):
                    continue

                if theme:
                    if theme != last_theme:
                        last_project = ""
                        last_subitem = ""
                    last_theme = theme
                else:
                    theme = last_theme

                if project:
                    last_project = project
                else:
                    project = last_project

                if subitem in {"", "-"}:
                    subitem = last_subitem
                elif subitem in {"子项", "分项", "细分项", "项目"}:
                    subitem = detail if detail else last_subitem
                else:
                    last_subitem = subitem

                if detail in {"", "-"}:
                    detail_value: Optional[str] = None
                else:
                    detail_value = detail

                values: Dict[int, float] = {}
                for year in YEARS:
                    text = (raw.get(str(year)) or "").strip()
                    if text and text != "-":
                        values[year] = float(text)

                if not values:
                    continue

                self.rows.append(
                    DataRow(
                        theme=theme,
                        project=project,
                        subitem=subitem,
                        detail=detail_value,
                        unit=unit,
                        values=values,
                    )
                )

    def find_all(
        self,
        *,
        theme: Optional[str] = None,
        project: Optional[str] = None,
        subitem: Optional[str] = None,
        detail: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> List[DataRow]:
        results: List[DataRow] = []
        for row in self.rows:
            if theme is not None and row.theme != theme:
                continue
            if project is not None and row.project != project:
                continue
            if subitem is not None and row.subitem != subitem:
                continue
            if detail is not None and row.detail != detail:
                continue
            if unit is not None and row.unit != unit:
                continue
            results.append(row)
        return results

    def find_one(
        self,
        *,
        theme: Optional[str] = None,
        project: Optional[str] = None,
        subitem: Optional[str] = None,
        detail: Optional[str] = None,
        unit: Optional[str] = None,
    ) -> DataRow:
        matches = self.find_all(theme=theme, project=project, subitem=subitem, detail=detail, unit=unit)
        if not matches:
            criteria = {"theme": theme, "project": project, "subitem": subitem, "detail": detail, "unit": unit}
            raise KeyError(f"No row matches {criteria}")
        if len(matches) > 1:
            criteria = {"theme": theme, "project": project, "subitem": subitem, "detail": detail, "unit": unit}
            raise ValueError(f"Multiple rows match {criteria}")
        return matches[0]


def percent_change(series: List[float]) -> Dict[int, float]:
    changes: Dict[int, float] = {}
    for idx in range(1, len(series)):
        prev = series[idx - 1]
        curr = series[idx]
        year = YEARS[idx]
        if prev == 0:
            changes[year] = math.nan
        else:
            changes[year] = (curr - prev) / prev
    return changes


def log_mean(a: float, b: float) -> float:
    if a == b:
        return a
    if a > 0 and b > 0:
        return (a - b) / (math.log(a) - math.log(b))
    raise ValueError("Log mean undefined for non-positive numbers")


def lmdi_decomposition(
    c_start: float,
    c_end: float,
    *,
    population_start: float,
    population_end: float,
    gdp_per_cap_start: float,
    gdp_per_cap_end: float,
    energy_intensity_start: float,
    energy_intensity_end: float,
    carbon_intensity_start: float,
    carbon_intensity_end: float,
) -> Dict[str, float]:
    weight = log_mean(c_end, c_start)
    contributions = {
        "population": weight * math.log(population_end / population_start),
        "gdp_per_capita": weight * math.log(gdp_per_cap_end / gdp_per_cap_start),
        "energy_intensity": weight * math.log(energy_intensity_end / energy_intensity_start),
        "carbon_intensity": weight * math.log(carbon_intensity_end / carbon_intensity_start),
    }
    return contributions


def average_annual_growth(start: float, end: float, years_count: int) -> float:
    if start <= 0 or end <= 0:
        raise ValueError("AAGR requires positive values")
    return (end / start) ** (1 / years_count) - 1


def ensure_dirs() -> None:
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("figures", exist_ok=True)


def write_svg_bar_chart(series_map: Dict[str, List[float]], years: List[int], path: str) -> None:
    width = 1200
    height = 800
    margin = 60
    cols = 2
    rows = 2
    chart_width = (width - (cols + 1) * margin) / cols
    chart_height = (height - (rows + 1) * margin) / rows
    bar_padding = 20

    svg_parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:"Arial";font-size:14px;fill:#222} .title{font-size:18px;font-weight:bold}</style>',
    ]

    items = list(series_map.items())
    for idx, (title, values) in enumerate(items):
        col = idx % cols
        row = idx // cols
        x0 = margin + col * (chart_width + margin)
        y0 = margin + row * (chart_height + margin)
        plot_width = chart_width - bar_padding * 2
        plot_height = chart_height - bar_padding * 2
        max_value = max(values)
        if max_value == 0:
            scale = 0
        else:
            scale = plot_height / max_value

        # Background and title
        svg_parts.append(f'<rect x="{x0}" y="{y0}" width="{chart_width}" height="{chart_height}" fill="#f9f9f9" stroke="#ccc"/>')
        svg_parts.append(
            f'<text class="title" x="{x0 + chart_width / 2}" y="{y0 + 24}" text-anchor="middle">{title}</text>'
        )

        # Axes
        axis_x = y0 + chart_height - bar_padding
        axis_y = x0 + bar_padding
        svg_parts.append(
            f'<line x1="{axis_y}" y1="{axis_x}" x2="{axis_y + plot_width}" y2="{axis_x}" stroke="#444" stroke-width="1" />'
        )
        svg_parts.append(
            f'<line x1="{axis_y}" y1="{y0 + bar_padding}" x2="{axis_y}" y2="{axis_x}" stroke="#444" stroke-width="1" />'
        )

        bar_width = plot_width / len(years) * 0.6
        step = plot_width / len(years)
        for i, year in enumerate(years):
            value = values[i]
            bar_height = value * scale
            x = axis_y + i * step + (step - bar_width) / 2
            y = axis_x - bar_height
            svg_parts.append(
                f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="#4a90e2" />'
            )
            if len(years) <= 11 or i % 2 == 0:
                svg_parts.append(
                    f'<text x="{x + bar_width / 2}" y="{axis_x + 18}" text-anchor="middle">{year}</text>'
                )
        svg_parts.append(
            f'<text x="{axis_y}" y="{y0 + bar_padding - 8}" text-anchor="start">最大值: {max_value:.0f}</text>'
        )

    svg_parts.append('</svg>')
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(svg_parts))


def correlation_matrix(data: Dict[str, List[float]]) -> Dict[Tuple[str, str], float]:
    matrix: Dict[Tuple[str, str], float] = {}
    keys = list(data.keys())
    for i, key_i in enumerate(keys):
        for j in range(i, len(keys)):
            key_j = keys[j]
            corr = statistics.correlation(data[key_i], data[key_j])
            matrix[(key_i, key_j)] = corr
            matrix[(key_j, key_i)] = corr
    return matrix


def linear_regression_multi(features: List[List[float]], target: List[float]) -> Tuple[List[float], float]:
    """Ordinary least squares regression using normal equations."""

    n_samples = len(features)
    if n_samples != len(target):
        raise ValueError("Feature and target lengths differ")
    if n_samples == 0:
        raise ValueError("Empty dataset")

    n_features = len(features[0])
    # Assemble X^T X and X^T y with intercept
    xtx = [[0.0 for _ in range(n_features + 1)] for _ in range(n_features + 1)]
    xty = [0.0 for _ in range(n_features + 1)]

    for row, y in zip(features, target):
        extended = [1.0] + row
        for i in range(n_features + 1):
            xty[i] += extended[i] * y
            for j in range(n_features + 1):
                xtx[i][j] += extended[i] * extended[j]

    # Solve using Gaussian elimination
    coeffs = _solve_linear_system(xtx, xty)
    intercept = coeffs[0]
    weights = coeffs[1:]
    return weights, intercept


def _solve_linear_system(matrix: List[List[float]], vector: List[float]) -> List[float]:
    n = len(vector)
    # Augment the matrix with the vector
    aug = [row[:] + [vector[idx]] for idx, row in enumerate(matrix)]

    for col in range(n):
        # Find pivot
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            raise ValueError("Matrix is singular or ill-conditioned")
        # Swap
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        # Normalize pivot row
        pivot = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot
        # Eliminate
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]

    return [aug[i][n] for i in range(n)]


def main() -> None:
    ensure_dirs()
    econ_table = DataTable("经济与能源.csv")
    carbon_table = DataTable("碳排放.csv")

    population_row = econ_table.find_one(project="常驻人口", subitem="总量1")
    gdp_row = econ_table.find_one(project="GDP", subitem="总量")
    energy_total_row = econ_table.find_one(project="能源消费量", subitem="总量")
    carbon_total_row = carbon_table.find_one(project="碳排放量", subitem="总量")

    population = population_row.series()
    gdp = gdp_row.series()
    energy_total = energy_total_row.series()
    carbon_total = carbon_total_row.series()

    per_capita_gdp = [g / p for g, p in zip(gdp, population)]
    energy_intensity = [e / g for e, g in zip(energy_total, gdp)]
    carbon_intensity = [c / e for c, e in zip(carbon_total, energy_total)]

    # Non-fossil energy consumption (新能源热力 + 新能源电力 + 其他新能源)
    new_heat = econ_table.find_one(project="新能源热力", subitem="总量").series()
    new_power = econ_table.find_one(project="新能源电力", subitem="总量").series()
    other_new = econ_table.find_one(project="其他新能源", subitem="总量").series()
    non_fossil = [h + p + o for h, p, o in zip(new_heat, new_power, other_new)]
    non_fossil_share = [nf / e if e else 0.0 for nf, e in zip(non_fossil, energy_total)]

    macro_series = {
        "常住人口(万人)": population,
        "GDP(亿元)": gdp,
        "能源消费量(万tce)": energy_total,
        "碳排放量(万tCO2)": carbon_total,
    }
    write_svg_bar_chart(macro_series, YEARS, os.path.join("figures", "macro_indicators_bar.svg"))

    yoy = {
        "population": percent_change(population),
        "gdp": percent_change(gdp),
        "energy": percent_change(energy_total),
        "carbon": percent_change(carbon_total),
        "per_capita_gdp": percent_change(per_capita_gdp),
        "energy_intensity": percent_change(energy_intensity),
        "carbon_intensity": percent_change(carbon_intensity),
        "non_fossil_share": percent_change(non_fossil_share),
    }

    # Ring ratios identical to YoY for annual data but stored separately for clarity
    ring_ratio = {name: dict(values) for name, values in yoy.items()}

    # Sectoral carbon data
    sector_names = {
        "能源供应部门": ("第二产业", "能源供应部门"),
        "工业消费部门": ("第二产业", "工业消费部门"),
        "建筑消费部门": ("第三产业", "建筑消费部门"),
        "交通消费部门": ("第三产业", "交通消费部门"),
        "居民生活消费": ("居民生活", "居民生活消费"),
        "农林消费部门": ("第一产业", "农林消费部门"),
    }

    sector_carbon: Dict[str, List[float]] = {}
    for label, (project, subitem) in sector_names.items():
        rows = carbon_table.find_all(project=project, subitem=subitem)
        if rows:
            sector_carbon[label] = rows[0].series()
        else:
            sector_carbon[label] = [0.0 for _ in YEARS]

    # Energy supply department missing in carbon table -> compute via energy consumption and emission factors
    if all(all(value == 0 for value in values) for values in [sector_carbon["能源供应部门"]]):
        # Build energy consumption by fuel for energy supply
        supply_energy_rows = [
            row
            for row in econ_table.find_all(project="第二产业", subitem="能源供应部门")
            if row.unit == "万tce" and row.detail
        ]
        supply_carbon = [0.0 for _ in YEARS]
        if supply_energy_rows:
            fuels = ["煤炭", "油品", "天然气", "其他能源"]
            factor_lookup: Dict[str, List[float]] = {}
            for fuel in fuels:
                try:
                    factor_row = carbon_table.find_one(
                        theme="能源供应部门碳排放因子4", project="发电", detail=fuel
                    )
                    factor_lookup[fuel] = factor_row.series()
                except KeyError:
                    continue
            for row in supply_energy_rows:
                if row.detail in factor_lookup:
                    factors = factor_lookup[row.detail]
                    consumption = row.series()
                    for idx in range(len(YEARS)):
                        value = consumption[idx]
                        if value >= 0:
                            supply_carbon[idx] += value * factors[idx]
        sector_carbon["能源供应部门"] = supply_carbon

    # Summary statistics
    summary = {
        "years": YEARS,
        "population": population,
        "gdp": gdp,
        "energy_total": energy_total,
        "carbon_total": carbon_total,
        "per_capita_gdp": per_capita_gdp,
        "energy_intensity": energy_intensity,
        "carbon_intensity": carbon_intensity,
        "non_fossil_share": non_fossil_share,
        "sector_carbon": sector_carbon,
    }

    with open(os.path.join("outputs", "indicator_series.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    with open(os.path.join("outputs", "yoy_growth.json"), "w", encoding="utf-8") as fh:
        json.dump({k: {str(year): v for year, v in sorted(values.items())} for k, values in yoy.items()}, fh, ensure_ascii=False, indent=2)
    with open(os.path.join("outputs", "ring_ratio.json"), "w", encoding="utf-8") as fh:
        json.dump({k: {str(year): v for year, v in sorted(values.items())} for k, values in ring_ratio.items()}, fh, ensure_ascii=False, indent=2)

    # Correlation analysis among main indicators
    corr_inputs = {
        "碳排放量": carbon_total,
        "能源消费量": energy_total,
        "GDP": gdp,
        "常住人口": population,
        "人均GDP": per_capita_gdp,
        "单位GDP能耗": energy_intensity,
        "单位能耗碳排放": carbon_intensity,
        "非化石能源比重": non_fossil_share,
    }
    corr_matrix = correlation_matrix(corr_inputs)
    with open(os.path.join("outputs", "correlation_matrix.json"), "w", encoding="utf-8") as fh:
        json.dump({f"{a}|{b}": value for (a, b), value in corr_matrix.items()}, fh, ensure_ascii=False, indent=2)

    # LMDI decomposition for 2010-2015 and 2015-2020
    periods = {"2010-2015": (2010, 2015), "2015-2020": (2015, 2020)}
    lmdi_results: Dict[str, Dict[str, float]] = {}
    for label, (start_year, end_year) in periods.items():
        start_idx = YEARS.index(start_year)
        end_idx = YEARS.index(end_year)
        result = lmdi_decomposition(
            carbon_total[start_idx],
            carbon_total[end_idx],
            population_start=population[start_idx],
            population_end=population[end_idx],
            gdp_per_cap_start=per_capita_gdp[start_idx],
            gdp_per_cap_end=per_capita_gdp[end_idx],
            energy_intensity_start=energy_intensity[start_idx],
            energy_intensity_end=energy_intensity[end_idx],
            carbon_intensity_start=carbon_intensity[start_idx],
            carbon_intensity_end=carbon_intensity[end_idx],
        )
        result["total_change"] = carbon_total[end_idx] - carbon_total[start_idx]
        result["aagr"] = average_annual_growth(carbon_total[start_idx], carbon_total[end_idx], end_year - start_year)
        lmdi_results[label] = result

    with open(os.path.join("outputs", "lmdi_decomposition.json"), "w", encoding="utf-8") as fh:
        json.dump(lmdi_results, fh, ensure_ascii=False, indent=2)

    # Multi-variable regression: carbon vs (GDP, energy, population)
    features = [[g, e, p] for g, e, p in zip(gdp, energy_total, population)]
    weights, intercept = linear_regression_multi(features, carbon_total)
    regression = {
        "weights": {
            "GDP": weights[0],
            "能源消费量": weights[1],
            "常住人口": weights[2],
        },
        "intercept": intercept,
    }
    with open(os.path.join("outputs", "carbon_regression.json"), "w", encoding="utf-8") as fh:
        json.dump(regression, fh, ensure_ascii=False, indent=2)

    # Forecast parameter suggestions based on 2015-2020 averages
    def _avg_recent(data: Dict[int, float], start_year: int = 2016) -> float:
        values = [value for year, value in data.items() if year >= start_year]
        return sum(values) / len(values) if values else 0.0

    params = {
        "energy_intensity_reduction": _avg_recent(yoy["energy_intensity"]),
        "carbon_intensity_reduction": _avg_recent(yoy["carbon_intensity"]),
        "non_fossil_share_increase": _avg_recent(yoy["non_fossil_share"]),
        "gdp_growth": _avg_recent(yoy["gdp"]),
        "population_growth": _avg_recent(yoy["population"]),
    }
    with open(os.path.join("outputs", "forecast_parameters.json"), "w", encoding="utf-8") as fh:
        json.dump(params, fh, ensure_ascii=False, indent=2)

    # Human-readable summary for reporting
    summary_lines = []
    summary_lines.append("# 指标与分析结果摘要\n")
    summary_lines.append("## 宏观指标概览")
    summary_lines.append(
        f"2010-2020年常住人口从{population[0]:,.2f}万人增至{population[-1]:,.2f}万人，GDP由{gdp[0]:,.2f}亿元增至{gdp[-1]:,.2f}亿元。"
    )
    summary_lines.append(
        f"能源消费量由{energy_total[0]:,.2f}万tce升至{energy_total[-1]:,.2f}万tce，碳排放量从{carbon_total[0]:,.2f}万tCO₂增至{carbon_total[-1]:,.2f}万tCO₂。"
    )
    summary_lines.append("## 指标效率与结构")
    summary_lines.append(
        f"人均GDP在十年间提升至{per_capita_gdp[-1]:.2f}万元/人；单位GDP能耗由{energy_intensity[0]:.3f}降至{energy_intensity[-1]:.3f}万tce/亿元。"
    )
    summary_lines.append(
        f"单位能耗碳排放由{carbon_intensity[0]:.3f}降至{carbon_intensity[-1]:.3f} tCO₂/tce，非化石能源消费比重提升至{non_fossil_share[-1]*100:.2f}%。"
    )
    summary_lines.append("## 部门碳排放")
    for sector, values in sector_carbon.items():
        summary_lines.append(
            f"{sector}碳排放在2020年为{values[-1]:,.2f}万tCO₂，相比2010年的{values[0]:,.2f}万tCO₂变化{values[-1] - values[0]:+.2f}万tCO₂。"
        )
    summary_lines.append("## LMDI分解")
    for label, result in lmdi_results.items():
        summary_lines.append(
            f"{label}期间碳排放变化{result['total_change']:+.2f}万tCO₂，人口、经济发展、能耗强度、碳排放强度贡献分别为"
            f" {result['population']:+.2f}、{result['gdp_per_capita']:+.2f}、{result['energy_intensity']:+.2f}、{result['carbon_intensity']:+.2f}万tCO₂。"
        )
    summary_lines.append("## 多元回归模型")
    summary_lines.append(
        "碳排放量与GDP、能源消费量、常住人口的回归模型系数如下："
        f" GDP系数{weights[0]:.4f}，能源消费系数{weights[1]:.4f}，人口系数{weights[2]:.4f}，截距{intercept:.2f}。"
    )

    with open(os.path.join("outputs", "analysis_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(summary_lines))


if __name__ == "__main__":
    main()
