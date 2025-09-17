#!/usr/bin/env python3
"""基于区域经济与能源数据的碳排放预测脚本。

该脚本读取《经济与能源.csv》和《碳排放.csv》两个数据文件，
构建满足以下三项要求的碳排放预测模型：

1. 碳排放量与人口、GDP 与能源消费总量相关联；
2. 碳排放量与各能源消费部门（工业、建筑、交通、居民、农林、能源供应）以及能源供应部门的能源消费量相关联；
3. 碳排放量与各能源消费部门以及能源供应部门的能源消费品种结构相关联。

脚本无需依赖第三方库，仅使用标准库实现多元线性回归、趋势预测
以及折线图（SVG）绘制，支持中文标题与图例。
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

YEARS: List[int] = list(range(2010, 2021))
YEAR_STRINGS: List[str] = [str(year) for year in YEARS]
FUTURE_YEARS: List[int] = list(range(2021, 2026))
FILL_FIELDS: Tuple[str, ...] = ("主题", "项目", "子项", "单位")
ENERGY_TYPES: Tuple[str, ...] = ("煤炭", "油品", "天然气", "热力", "电力", "其他能源")
SVG_COLORS: List[str] = [
    "#4472c4",
    "#ed7d31",
    "#a5a5a5",
    "#ffc000",
    "#70ad47",
    "#5b9bd5",
    "#c00000",
    "#7030a0",
]
LEGEND_COLUMNS = 4


def load_and_fill_csv(path: str) -> List[Dict[str, str]]:
    """读取 CSV 并向下填充表头信息。"""
    rows: List[Dict[str, str]] = []
    last_values: Dict[str, str] = {field: "" for field in FILL_FIELDS}
    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for raw_row in reader:
            row = {key: (value.strip() if isinstance(value, str) else "") for key, value in raw_row.items()}
            if all(not row[field] for field in FILL_FIELDS) and all(not row.get(year, "") for year in YEAR_STRINGS):
                last_values = {field: "" for field in FILL_FIELDS}
                continue
            for field in FILL_FIELDS:
                if row.get(field):
                    last_values[field] = row[field]
                else:
                    row[field] = last_values[field]
            rows.append(row)
    return rows


def parse_float(text: Optional[str]) -> Optional[float]:
    """解析数值字符串，返回浮点数或 None。"""
    if text is None:
        return None
    stripped = text.strip()
    if not stripped or stripped == "-":
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


@dataclass
class LinearModel:
    """简单线性回归模型，支持岭回归正则项。"""

    feature_names: Sequence[str]
    alpha: float = 0.0
    coefficients: Optional[List[float]] = None

    def fit(self, features: Sequence[Sequence[float]], targets: Sequence[float]) -> None:
        design = [[1.0] + list(row) for row in features]
        transpose = list(zip(*design))
        xtx = [
            [sum(transpose[i][k] * transpose[j][k] for k in range(len(design))) for j in range(len(transpose))]
            for i in range(len(transpose))
        ]
        xty = [sum(transpose[i][k] * targets[k] for k in range(len(design))) for i in range(len(transpose))]
        if self.alpha > 0.0:
            for index in range(1, len(xtx)):
                xtx[index][index] += self.alpha
        self.coefficients = solve_linear_system(xtx, xty)

    def predict(self, features: Sequence[Sequence[float]]) -> List[float]:
        if self.coefficients is None:
            raise ValueError("模型尚未训练")
        preds: List[float] = []
        for row in features:
            value = self.coefficients[0]
            for idx, feature in enumerate(row):
                value += self.coefficients[idx + 1] * feature
            preds.append(value)
        return preds

    def r_squared(self, actual: Sequence[float], predicted: Sequence[float]) -> float:
        mean_actual = sum(actual) / len(actual)
        ss_total = sum((value - mean_actual) ** 2 for value in actual)
        ss_res = sum((value - pred) ** 2 for value, pred in zip(actual, predicted))
        if ss_total == 0:
            return 0.0
        return 1 - ss_res / ss_total

    def coefficient_table(self) -> List[Tuple[str, float]]:
        if self.coefficients is None:
            raise ValueError("模型尚未训练")
        pairs: List[Tuple[str, float]] = [("截距", self.coefficients[0])]
        for name, coef in zip(self.feature_names, self.coefficients[1:]):
            pairs.append((name, coef))
        return pairs


def solve_linear_system(matrix: List[List[float]], values: Sequence[float]) -> List[float]:
    """通过高斯-约当消元法求解线性方程组。"""
    n = len(matrix)
    augmented = [row[:] + [values[idx]] for idx, row in enumerate(matrix)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        if abs(augmented[pivot_row][col]) < 1e-12:
            raise ValueError("矩阵奇异，无法求解")
        if pivot_row != col:
            augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]
        pivot = augmented[col][col]
        for j in range(col, n + 1):
            augmented[col][j] /= pivot
        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                augmented[row][j] -= factor * augmented[col][j]
    return [augmented[i][n] for i in range(n)]


def fit_linear_trend(years: Sequence[int], values: Sequence[float]) -> Tuple[float, float]:
    """对单变量序列拟合线性趋势，返回截距与斜率。"""
    count = len(years)
    mean_year = sum(years) / count
    mean_value = sum(values) / count
    numerator = sum((year - mean_year) * (value - mean_value) for year, value in zip(years, values))
    denominator = sum((year - mean_year) ** 2 for year in years)
    if denominator == 0:
        return mean_value, 0.0
    slope = numerator / denominator
    intercept = mean_value - slope * mean_year
    return intercept, slope


def forecast_columns(matrix: Sequence[Sequence[float]], future_years: Sequence[int]) -> List[List[float]]:
    """按列对特征矩阵进行线性外推，返回未来年份的特征矩阵。"""
    columns = list(zip(*matrix))
    forecasts: List[List[float]] = []
    for column in columns:
        intercept, slope = fit_linear_trend(YEARS, list(column))
        forecasts.append([intercept + slope * year for year in future_years])
    result: List[List[float]] = []
    for idx in range(len(future_years)):
        row = [forecasts[col_idx][idx] for col_idx in range(len(columns))]
        result.append(row)
    return result


class DataRepository:
    """封装经济与碳排放数据的读取接口。"""

    def __init__(self, econ_path: str, carbon_path: str) -> None:
        self.econ_rows = load_and_fill_csv(econ_path)
        self.carbon_rows = load_and_fill_csv(carbon_path)

    def get_series(
        self,
        dataset: str,
        *,
        theme: Optional[str] = None,
        project: Optional[str] = None,
        subitem: Optional[str] = None,
        unit: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> List[float]:
        rows = self.econ_rows if dataset == "econ" else self.carbon_rows
        for row in rows:
            if theme is not None and row.get("主题") != theme:
                continue
            if project is not None and row.get("项目") != project:
                continue
            if subitem is not None and row.get("子项") != subitem:
                continue
            if unit is not None and row.get("单位") != unit:
                continue
            if detail is not None and row.get("细分项") != detail:
                continue
            values = [parse_float(row.get(year)) for year in YEAR_STRINGS]
            if any(value is None for value in values):
                continue
            return [value for value in values if value is not None]
        raise ValueError(
            f"未找到满足条件的行: dataset={dataset}, theme={theme}, project={project}, subitem={subitem}, unit={unit}, detail={detail}"
        )

    def get_sector_total(self, sector: str) -> List[float]:
        mapping = {
            "农林消费部门": ("能源消费量", "第一产业", "农林消费部门"),
            "工业消费部门": ("能源消费量", "第二产业", "工业消费部门"),
            "交通消费部门": ("能源消费量", "第三产业", "交通消费部门"),
            "建筑消费部门": ("能源消费量", "第三产业", "建筑消费部门"),
            "居民生活消费": ("能源消费量", "居民生活", "居民生活消费"),
        }
        if sector not in mapping:
            raise ValueError(f"不支持的部门: {sector}")
        theme, project, subitem = mapping[sector]
        return self.get_series("econ", theme=theme, project=project, subitem=subitem, unit="万tce", detail="-")

    def get_energy_mix_by_sector(self, sector: str) -> Dict[str, List[float]]:
        mixes: Dict[str, List[float]] = {}
        for row in self.econ_rows:
            if row.get("主题") != "产业能耗结构":
                continue
            if row.get("子项") != sector:
                continue
            if row.get("单位") != "万tce":
                continue
            kind = row.get("细分项")
            if not kind or kind == "-":
                continue
            mixes[kind] = [parse_float(row.get(year)) or 0.0 for year in YEAR_STRINGS]
        return mixes

    def get_energy_supply_mix(self) -> Dict[str, List[float]]:
        return self.get_energy_mix_by_sector("能源供应部门")

    def get_carbon_sector(self, sector: str) -> List[float]:
        mapping: Dict[str, Tuple[Optional[str], Optional[str], str, Optional[str]]] = {
            "农林消费部门": ("碳排放量1", "第一产业", "农林消费部门", "万tCO2"),
            "工业消费部门": ("碳排放量1", "第二产业", "工业消费部门", "万tCO2"),
            "交通消费部门": ("碳排放量1", "第三产业", "交通消费部门", "万tCO2"),
            "建筑消费部门": ("碳排放量1", "第三产业", "建筑消费部门", "万tCO2"),
            "第三产业总量": ("碳排放量1", "第三产业", "总量", "万tCO2"),
            "居民生活消费": ("碳排放量1", "居民生活", "居民生活消费", "万tCO2"),
        }
        theme, project, subitem, unit = mapping[sector]
        return self.get_series(
            "carbon",
            theme=theme,
            project=project,
            subitem=subitem,
            unit=unit,
            detail="-",
        )


def sum_vectors(vectors: Iterable[List[float]]) -> List[float]:
    totals = [0.0 for _ in YEAR_STRINGS]
    for vector in vectors:
        for idx, value in enumerate(vector):
            totals[idx] += value
    return totals


def normalize_index(series: Sequence[float]) -> List[float]:
    base = series[0]
    return [value / base * 100 if base else 0.0 for value in series]


def extend_with_none(series: Sequence[float], extra_count: int) -> List[Optional[float]]:
    extended = list(series)
    extended.extend([None] * extra_count)
    return extended


def combine_series(past: Sequence[float], future: Sequence[float]) -> List[float]:
    combined = list(past)
    combined.extend(future)
    return combined


def create_line_chart(
    path: str,
    title: str,
    series: Sequence[Tuple[str, Sequence[Optional[float]]]],
    years: Sequence[int],
    y_label: str,
) -> None:
    width, height = 900, 540
    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 80, 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    numeric_values = [value for _, seq in series for value in seq if value is not None]
    if not numeric_values:
        raise ValueError("系列数据为空，无法绘图")

    min_value = min(numeric_values)
    max_value = max(numeric_values)
    if math.isclose(max_value, min_value):
        max_value += 1.0
        min_value -= 1.0
    padding = (max_value - min_value) * 0.1
    y_min = min_value - padding
    y_max = max_value + padding
    x_step = plot_width / (len(years) - 1 if len(years) > 1 else 1)

    lines: List[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<style>text{font-family:'SimHei, Noto Sans CJK, sans-serif';font-size:16px;}</style>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' stroke='none'/>",
        f"<text x='{width/2}' y='{margin_top/2}' text-anchor='middle' font-size='22'>{title}</text>",
        f"<text x='{margin_left/2}' y='{margin_top + plot_height/2}' transform='rotate(-90 {margin_left/2} {margin_top + plot_height/2})' text-anchor='middle'>{y_label}</text>",
    ]

    axis_y = margin_top + plot_height
    lines.append(f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{axis_y}' stroke='black' stroke-width='1'/>")
    lines.append(f"<line x1='{margin_left}' y1='{axis_y}' x2='{margin_left + plot_width}' y2='{axis_y}' stroke='black' stroke-width='1'/>")

    for idx, year in enumerate(years):
        x = margin_left + idx * x_step
        lines.append(f"<line x1='{x}' y1='{axis_y}' x2='{x}' y2='{axis_y + 6}' stroke='black' stroke-width='1'/>")
        lines.append(f"<text x='{x}' y='{axis_y + 28}' text-anchor='middle'>{year}</text>")

    tick_count = 6
    for tick in range(tick_count + 1):
        ratio = tick / tick_count
        value = y_min + (y_max - y_min) * ratio
        y = axis_y - ratio * plot_height
        lines.append(f"<line x1='{margin_left}' y1='{y}' x2='{margin_left + plot_width}' y2='{y}' stroke='#dddddd' stroke-width='1'/>")
        lines.append(f"<text x='{margin_left - 10}' y='{y + 6}' text-anchor='end'>{value:.1f}</text>")

    for idx, (label, values) in enumerate(series):
        color = SVG_COLORS[idx % len(SVG_COLORS)]
        segment_points: List[str] = []
        circle_positions: List[Tuple[float, float]] = []
        for jdx, value in enumerate(values):
            x = margin_left + jdx * x_step
            if value is None:
                if len(segment_points) >= 2:
                    lines.append(f"<polyline fill='none' stroke='{color}' stroke-width='2' points={' '.join(segment_points)} />")
                segment_points = []
                continue
            y = axis_y - (value - y_min) / (y_max - y_min) * plot_height
            segment_points.append(f"{x},{y}")
            circle_positions.append((x, y))
        if len(segment_points) >= 2:
            lines.append(f"<polyline fill='none' stroke='{color}' stroke-width='2' points={' '.join(segment_points)} />")
        for cx, cy in circle_positions:
            lines.append(f"<circle cx='{cx}' cy='{cy}' r='3.5' fill='{color}' />")

    legend_base_y = margin_top - 35
    for idx, (label, _) in enumerate(series):
        row = idx // LEGEND_COLUMNS
        col = idx % LEGEND_COLUMNS
        lx = margin_left + col * 180
        ly = legend_base_y - row * 28
        color = SVG_COLORS[idx % len(SVG_COLORS)]
        lines.append(f"<rect x='{lx}' y='{ly - 12}' width='20' height='20' fill='{color}' />")
        lines.append(f"<text x='{lx + 30}' y='{ly + 5}'>{label}</text>")

    lines.append("</svg>")
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def ensure_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def compute_secondary_share(mix: Dict[str, List[float]], *, absolute: bool = False) -> List[float]:
    shares: List[float] = []
    for idx in range(len(YEARS)):
        total = 0.0
        secondary = 0.0
        for kind in ENERGY_TYPES:
            values = mix.get(kind, [0.0] * len(YEARS))
            value = values[idx]
            contribution = abs(value) if absolute else value
            total += contribution
            if kind in ("热力", "电力"):
                secondary += abs(value) if absolute else value
        shares.append(secondary / total * 100 if total else 0.0)
    return shares


def main() -> None:
    repo = DataRepository("经济与能源.csv", "碳排放.csv")
    output_dir = os.path.join("outputs", "emission_models")
    ensure_directory(output_dir)

    carbon_total = repo.get_series("carbon", theme="碳排放量1", project="碳排放量", subitem="总量", unit="万tCO2", detail="-")

    population = repo.get_series("econ", theme="人口", project="常驻人口", subitem="总量1", unit="万人", detail="-")
    gdp = repo.get_series("econ", theme="生产总值", project="GDP", subitem="总量", unit="亿元", detail="-")
    energy_total = repo.get_series("econ", theme="能源消费量", project="能源消费量", subitem="总量", unit="万tce", detail="-")

    model1_features = [list(row) for row in zip(population, gdp, energy_total)]
    model1 = LinearModel(["常住人口", "地区生产总值", "能源消费总量"], alpha=1e-4)
    model1.fit(model1_features, carbon_total)
    model1_predictions = model1.predict(model1_features)
    model1_r2 = model1.r_squared(carbon_total, model1_predictions)
    future_macro = forecast_columns(model1_features, FUTURE_YEARS)
    future_macro_preds = model1.predict(future_macro)
    model1_full_pred = combine_series(model1_predictions, future_macro_preds)

    sectors = [
        ("农林消费部门", "农林"),
        ("工业消费部门", "工业"),
        ("交通消费部门", "交通"),
        ("建筑消费部门", "建筑"),
        ("居民生活消费", "居民"),
    ]
    sector_totals = {key: repo.get_sector_total(key) for key, _ in sectors}
    supply_mix = repo.get_energy_supply_mix()
    supply_total = sum_vectors([supply_mix.get(kind, [0.0] * len(YEARS)) for kind in ("煤炭", "油品", "天然气")])

    model2_rows: List[List[float]] = []
    for idx in range(len(YEARS)):
        row = [sector_totals[key][idx] for key, _ in sectors]
        row.append(supply_total[idx])
        model2_rows.append(row)
    model2 = LinearModel(
        ["农林能源消费", "工业能源消费", "交通能源消费", "建筑能源消费", "居民能源消费", "能源供应化石消耗"],
        alpha=1e-4,
    )
    model2.fit(model2_rows, carbon_total)
    model2_predictions = model2.predict(model2_rows)
    model2_r2 = model2.r_squared(carbon_total, model2_predictions)
    future_model2 = forecast_columns(model2_rows, FUTURE_YEARS)
    future_model2_preds = model2.predict(future_model2)
    model2_full_pred = combine_series(model2_predictions, future_model2_preds)

    sector_mix_data = {key: repo.get_energy_mix_by_sector(key) for key, _ in sectors}
    supply_mix_data = repo.get_energy_supply_mix()
    model3_rows: List[List[float]] = []
    for idx in range(len(YEARS)):
        row: List[float] = []
        for key, _ in sectors:
            mix = sector_mix_data[key]
            fossil = sum(mix.get(kind, [0.0] * len(YEARS))[idx] for kind in ("煤炭", "油品", "天然气"))
            secondary = sum(mix.get(kind, [0.0] * len(YEARS))[idx] for kind in ("热力", "电力"))
            non_fossil = mix.get("其他能源", [0.0] * len(YEARS))[idx]
            row.extend([fossil, secondary, non_fossil])
        supply_fossil = sum(supply_mix_data.get(kind, [0.0] * len(YEARS))[idx] for kind in ("煤炭", "油品", "天然气"))
        supply_secondary = sum(supply_mix_data.get(kind, [0.0] * len(YEARS))[idx] for kind in ("热力", "电力"))
        supply_non_fossil = supply_mix_data.get("其他能源", [0.0] * len(YEARS))[idx]
        row.extend([supply_fossil, supply_secondary, supply_non_fossil])
        model3_rows.append(row)

    feature_names_model3: List[str] = []
    for _, label in sectors:
        feature_names_model3.extend([f"{label}化石能源", f"{label}二次能源", f"{label}其他能源"])
    feature_names_model3.extend(["能源供应化石投入", "能源供应二次能源", "能源供应其他能源"])

    model3 = LinearModel(feature_names_model3, alpha=1e-2)
    model3.fit(model3_rows, carbon_total)
    model3_predictions = model3.predict(model3_rows)
    model3_r2 = model3.r_squared(carbon_total, model3_predictions)
    future_model3 = forecast_columns(model3_rows, FUTURE_YEARS)
    future_model3_preds = model3.predict(future_model3)
    model3_full_pred = combine_series(model3_predictions, future_model3_preds)

    carbon_extended = extend_with_none(carbon_total, len(FUTURE_YEARS))

    create_line_chart(
        os.path.join(output_dir, "model1_predictions.svg"),
        "模型一：宏观变量驱动的碳排放预测",
        [
            ("实际碳排放", carbon_extended),
            ("模型拟合及预测", model1_full_pred),
        ],
        YEARS + FUTURE_YEARS,
        "碳排放量（万tCO₂）",
    )

    macro_indices = [
        ("常住人口指数", normalize_index(population)),
        ("GDP 指数", normalize_index(gdp)),
        ("能源消费指数", normalize_index(energy_total)),
        ("碳排放指数", normalize_index(carbon_total)),
    ]
    create_line_chart(
        os.path.join(output_dir, "macro_trends.svg"),
        "宏观驱动指标指数（2010=100）",
        macro_indices,
        YEARS,
        "指数（2010=100）",
    )

    create_line_chart(
        os.path.join(output_dir, "model2_predictions.svg"),
        "模型二：部门能源消费驱动的碳排放预测",
        [
            ("实际碳排放", carbon_extended),
            ("模型拟合及预测", model2_full_pred),
        ],
        YEARS + FUTURE_YEARS,
        "碳排放量（万tCO₂）",
    )

    energy_series = [(label + "能源消费", sector_totals[key]) for key, label in sectors]
    energy_series.append(("能源供应化石投入", supply_total))
    create_line_chart(
        os.path.join(output_dir, "sector_energy_consumption.svg"),
        "各部门能源消费总量",
        energy_series,
        YEARS,
        "能源消费量（万tce）",
    )

    carbon_series = [
        ("工业碳排放", repo.get_carbon_sector("工业消费部门")),
        ("交通碳排放", repo.get_carbon_sector("交通消费部门")),
        ("建筑碳排放", repo.get_carbon_sector("建筑消费部门")),
        ("居民碳排放", repo.get_carbon_sector("居民生活消费")),
        ("农林碳排放", repo.get_carbon_sector("农林消费部门")),
    ]
    create_line_chart(
        os.path.join(output_dir, "sector_carbon_emissions.svg"),
        "主要消费部门碳排放量",
        carbon_series,
        YEARS,
        "碳排放量（万tCO₂）",
    )

    secondary_shares = [
        (label + "二次能源占比(%)", compute_secondary_share(sector_mix_data[key]))
        for key, label in sectors
    ]
    secondary_shares.append(("能源供应二次能源占比(%)", compute_secondary_share(supply_mix_data, absolute=True)))
    create_line_chart(
        os.path.join(output_dir, "secondary_energy_share.svg"),
        "各部门二次能源（电与热）占比",
        secondary_shares,
        YEARS,
        "占比（%）",
    )

    create_line_chart(
        os.path.join(output_dir, "model3_predictions.svg"),
        "模型三：能源消费品种结构驱动的碳排放预测",
        [
            ("实际碳排放", carbon_extended),
            ("模型拟合及预测", model3_full_pred),
        ],
        YEARS + FUTURE_YEARS,
        "碳排放量（万tCO₂）",
    )

    metrics_path = os.path.join(output_dir, "model_summary.txt")
    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write("区域碳排放预测模型评估摘要\n")
        file.write("------------------------------------------------------------\n")
        file.write("模型一：宏观变量驱动\n")
        file.write(f"  决定系数 R² = {model1_r2:.4f}\n")
        file.write(f"  2025 年预测碳排放 = {future_macro_preds[-1]:.2f} 万tCO₂\n")
        file.write("  回归系数：\n")
        for name, coef in model1.coefficient_table():
            file.write(f"    {name}: {coef:.6f}\n")
        file.write("\n")

        file.write("模型二：部门能源消费驱动\n")
        file.write(f"  决定系数 R² = {model2_r2:.4f}\n")
        file.write(f"  2025 年预测碳排放 = {future_model2_preds[-1]:.2f} 万tCO₂\n")
        file.write("  回归系数：\n")
        for name, coef in model2.coefficient_table():
            file.write(f"    {name}: {coef:.6f}\n")
        file.write("\n")

        file.write("模型三：能源消费品种结构驱动\n")
        file.write(f"  决定系数 R² = {model3_r2:.4f}\n")
        file.write(f"  2025 年预测碳排放 = {future_model3_preds[-1]:.2f} 万tCO₂\n")
        file.write("  回归系数：\n")
        for name, coef in model3.coefficient_table():
            file.write(f"    {name}: {coef:.6f}\n")
        file.write("\n")
        file.write("说明：未来年份的驱动因素通过 2010-2020 年的线性趋势外推得到。\n")

    print("模型一（宏观变量）R²:", f"{model1_r2:.4f}", "2025预测:", f"{future_macro_preds[-1]:.2f} 万tCO₂")
    print("模型二（部门能源消费）R²:", f"{model2_r2:.4f}", "2025预测:", f"{future_model2_preds[-1]:.2f} 万tCO₂")
    print("模型三（能源品种结构）R²:", f"{model3_r2:.4f}", "2025预测:", f"{future_model3_preds[-1]:.2f} 万tCO₂")
    print("图表及评估摘要输出目录:", output_dir)


if __name__ == "__main__":
    main()
