import csv
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SCAN_START = date(2020, 1, 23)
SCAN_END = date(2023, 12, 31)
TRILLION = 1e12

SOCIAL_EVENTS = [
    (date(2020, 2, 7), 5, 5, 0.98),
    (date(2020, 7, 17), 3, 3, 0.90),
    (date(2021, 1, 6), 3, 3, 0.95),
    (date(2021, 7, 5), 4, 3, 0.90),
    (date(2021, 8, 3), 3, 3, 0.90),
    (date(2021, 12, 23), 4, 4, 0.95),
    (date(2022, 1, 4), 4, 4, 0.90),
    (date(2022, 3, 14), 3, 4, 0.95),
    (date(2022, 3, 28), 5, 5, 0.98),
    (date(2022, 4, 22), 4, 5, 0.90),
    (date(2022, 5, 11), 4, 5, 0.90),
    (date(2022, 9, 1), 3, 4, 0.95),
    (date(2022, 9, 5), 3, 4, 0.85),
    (date(2022, 9, 18), 5, 5, 0.98),
    (date(2022, 10, 13), 3, 5, 0.95),
    (date(2022, 10, 29), 4, 5, 0.90),
    (date(2022, 11, 1), 4, 4, 0.85),
    (date(2022, 11, 14), 3, 4, 0.90),
    (date(2022, 11, 23), 4, 5, 0.95),
    (date(2022, 11, 24), 5, 5, 0.98),
    (date(2022, 11, 26), 5, 5, 0.95),
    (date(2022, 11, 29), 4, 4, 0.90),
]


def interpolate(day, anchors):
    anchors = sorted(anchors)
    if day <= anchors[0][0]:
        return anchors[0][1]
    if day >= anchors[-1][0]:
        return anchors[-1][1]
    for (d0, v0), (d1, v1) in zip(anchors, anchors[1:]):
        if d0 <= day <= d1:
            f = (day - d0).days / (d1 - d0).days
            return v0 + f * (v1 - v0)
    raise RuntimeError("unreachable")


def projected_deaths(day):
    """Deaths after full release, including capacity pressure; persons.

    The March-2022 anchor is the 1.55 million estimate from Cai et al.
    (Nature Medicine, 2022). Other anchors are transparent scenario priors,
    not observed counterfactual outcomes.
    """
    return interpolate(day, [
        (date(2020, 1, 23), 8_600_000),
        (date(2020, 6, 1), 7_200_000),
        (date(2021, 1, 1), 5_800_000),
        (date(2021, 7, 1), 3_500_000),
        (date(2022, 3, 1), 1_550_000),
        (date(2022, 9, 1), 1_300_000),
        (date(2022, 12, 7), 1_050_000),
        (date(2023, 1, 15), 900_000),
        (date(2023, 3, 1), 850_000),
        (date(2023, 6, 1), 700_000),
        (date(2023, 12, 31), 600_000),
    ])


def projected_peak_capacity_ratio(day):
    """Projected peak critical-care demand divided by effective capacity.

    The March-2022 value is anchored to Cai et al.'s 15.6-fold estimate. The
    late-2022 path also reflects the rapid published capacity expansion. A
    value above one is treated as operationally infeasible in the base case.
    """
    return interpolate(day, [
        (date(2020, 1, 23), 28.0),
        (date(2021, 1, 1), 22.0),
        (date(2022, 3, 1), 15.6),
        (date(2022, 9, 1), 5.0),
        (date(2022, 11, 22), 2.5),
        (date(2022, 12, 7), 0.95),
        (date(2023, 1, 15), 0.68),
        (date(2023, 3, 1), 0.62),
        (date(2023, 6, 1), 0.52),
        (date(2023, 12, 31), 0.45),
    ])


def daily_control_cost(day):
    """Incremental containment cost in RMB/day relative to risk management.

    This is a macro scenario path, not a causal decomposition of observed GDP.
    It represents testing, isolation, mobility disruption, delayed care,
    education and compliance costs. The path is deliberately exposed for
    sensitivity analysis.
    """
    billion = 1e9
    return interpolate(day, [
        (date(2020, 1, 23), 22.0 * billion),
        (date(2020, 4, 1), 5.0 * billion),
        (date(2020, 7, 1), 1.2 * billion),
        (date(2021, 1, 1), 1.5 * billion),
        (date(2021, 7, 1), 2.0 * billion),
        (date(2022, 1, 1), 4.0 * billion),
        (date(2022, 4, 1), 18.0 * billion),
        (date(2022, 6, 1), 5.0 * billion),
        (date(2022, 10, 1), 10.0 * billion),
        (date(2022, 12, 7), 15.0 * billion),
        (date(2023, 3, 1), 16.0 * billion),
        (date(2023, 12, 31), 18.0 * billion),
    ])


def social_fatigue(day):
    """Baseline fatigue plus decaying, geographically weighted event impulses."""
    baseline = interpolate(day, [
        (date(2020, 1, 23), 0.05),
        (date(2021, 1, 1), 0.18),
        (date(2022, 3, 1), 0.40),
        (date(2022, 6, 1), 0.48),
        (date(2022, 10, 1), 0.58),
        (date(2022, 12, 7), 0.65),
        (date(2023, 3, 1), 0.75),
        (date(2023, 12, 31), 0.90),
    ])
    event_pressure = 0.0
    for event_day, severity, reach, reliability in SOCIAL_EVENTS:
        if day >= event_day:
            age = (day - event_day).days
            impulse = severity * reach * reliability
            event_pressure += (0.005 * impulse * np.exp(-age / 90.0)
                               + 0.001 * impulse * np.exp(-age / 540.0))
    risk_salience = interpolate(day, [
        (date(2020, 1, 23), 1.00),
        (date(2020, 6, 1), 0.75),
        (date(2021, 1, 1), 0.60),
        (date(2022, 3, 1), 0.35),
        (date(2022, 12, 7), 0.15),
        (date(2023, 12, 31), 0.10),
    ])
    return max(0.0, baseline + event_pressure - 0.12 * risk_salience)


def daily_governance_cost(day):
    """Convex social/governance loss once fatigue approaches its critical range."""
    excess = max(0.0, social_fatigue(day) - 0.72)
    return 220e9 * excess ** 2


def release_wave_cost(day):
    """Temporary output loss from illness, absenteeism and adjustment after release."""
    deaths_ratio = projected_deaths(day) / projected_deaths(date(2022, 12, 7))
    return (0.55 + 0.35 * min(deaths_ratio, 4.0)) * TRILLION


def control_cost_series(cost_scale=1.0):
    economic_total = 0.0
    governance_total = 0.0
    values = {}
    day = SCAN_START
    while day <= SCAN_END:
        values[day] = (economic_total, governance_total)
        economic_total += daily_control_cost(day) * cost_scale
        governance_total += daily_governance_cost(day) * cost_scale
        day += timedelta(days=1)
    return values


def evaluate_all(vsl=5_000_000.0, cost_scale=1.0, death_scale=1.0):
    cumulative = control_cost_series(cost_scale)
    rows = []
    day = SCAN_START
    while day <= SCAN_END:
        deaths = projected_deaths(day) * death_scale
        health_loss = deaths * vsl
        control_loss, governance_loss = cumulative[day]
        wave_loss = release_wave_cost(day) * cost_scale
        total = health_loss + control_loss + governance_loss + wave_loss
        rows.append({
            "date": day,
            "deaths": deaths,
            "health_loss": health_loss,
            "control_loss": control_loss,
            "governance_loss": governance_loss,
            "release_wave_loss": wave_loss,
            "total_loss": total,
            "peak_capacity_ratio": projected_peak_capacity_ratio(day),
        })
        day += timedelta(days=1)
    return rows


def optimum(rows, demand_scale=1.0, clustering_multiplier=1.20, capacity_limit=1.15):
    feasible = [i for i, row in enumerate(rows)
                if row["peak_capacity_ratio"] * demand_scale * clustering_multiplier <= capacity_limit]
    if not feasible:
        raise RuntimeError("No medically feasible reopening date in scan horizon")
    return min(feasible, key=lambda i: rows[i]["total_loss"])


def sensitivity():
    choices = []
    table = []
    for vsl in (2_000_000, 3_500_000, 5_000_000, 7_500_000, 10_000_000):
        dates = []
        for cost_scale in (0.70, 0.85, 1.00, 1.15, 1.30):
            for death_scale in (0.80, 1.00, 1.20):
                for demand_scale in (0.85, 1.00, 1.15):
                    for clustering_multiplier in (1.00, 1.20, 1.40):
                        rows = evaluate_all(vsl, cost_scale, death_scale)
                        chosen = rows[optimum(rows, demand_scale, clustering_multiplier)]["date"]
                        dates.append(chosen)
                        choices.append(chosen)
        dates.sort()
        table.append({
            "vsl_rmb": vsl,
            "p10": percentile_date(dates, 0.10),
            "median": percentile_date(dates, 0.50),
            "p90": percentile_date(dates, 0.90),
        })
    choices.sort()
    return choices, table


def percentile_date(values, p):
    return values[round((len(values) - 1) * p)]


def font(size, bold=False):
    path = Path("C:/Windows/Fonts") / ("msyhbd.ttc" if bold else "msyh.ttc")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def chart(rows, best_idx, low, high):
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = 150, 120, 80, 160
    pw, ph = w-left-right, h-top-bottom
    draw.text((left, 30), "Full-release date: economic and mortality trade-off",
              fill="#111827", font=font(36, True))
    draw.text((left, 78), "Default VSL = RMB 5 million; all losses in RMB trillion",
              fill="#4b5563", font=font(21))
    series = [
        ("Death loss", np.array([r["health_loss"] for r in rows]) / TRILLION, "#b91c1c"),
        ("Control + governance loss",
         np.array([r["control_loss"] + r["governance_loss"] for r in rows]) / TRILLION,
         "#0369a1"),
        ("Total loss", np.array([r["total_loss"] for r in rows]) / TRILLION, "#111827"),
    ]
    ymax = max(float(v.max()) for _, v, _ in series) * 1.04
    for j in range(6):
        y = top + ph*j/5
        draw.line((left, y, left+pw, y), fill="#e5e7eb", width=2)
        draw.text((35, y-13), f"{ymax*(1-j/5):.1f}", fill="#4b5563", font=font(19))
    low_i, high_i = (low-SCAN_START).days, (high-SCAN_START).days
    x0 = left + pw*low_i/(len(rows)-1)
    x1 = left + pw*high_i/(len(rows)-1)
    draw.rectangle((x0, top, x1, top+ph), fill="#dcfce7")
    for label, values, color in series:
        points = []
        for i, value in enumerate(values):
            x = left + pw*i/(len(rows)-1)
            y = top + ph*(1-value/ymax)
            points.append((x, y))
        draw.line(points, fill=color, width=5)
    bx = left + pw*best_idx/(len(rows)-1)
    best_y = top + ph*(1-series[2][1][best_idx]/ymax)
    draw.ellipse((bx-8,best_y-8,bx+8,best_y+8), fill="#7c3aed")
    draw.text((min(bx+15,w-410),best_y-34), rows[best_idx]["date"].isoformat(),
              fill="#6d28d9", font=font(21, True))
    ticks = [(date(y,1,1),str(y)) for y in (2021,2022,2023)]
    ticks.insert(0,(SCAN_START,"2020"))
    for day,label in ticks:
        i=(day-SCAN_START).days
        x=left+pw*i/(len(rows)-1)
        draw.line((x,top+ph,x,top+ph+8),fill="#374151",width=2)
        draw.text((x-25,top+ph+18),label,fill="#374151",font=font(20))
    lx=left
    for label,_,color in series:
        draw.line((lx,h-78,lx+38,h-78),fill=color,width=6)
        draw.text((lx+48,h-92),label,fill="#111827",font=font(19))
        lx += 390
    draw.text((left,h-125),f"Green: middle 80% sensitivity window ({low} to {high})",
              fill="#374151",font=font(19))
    img.save(OUT / "防疫模型_经济死亡权衡最优放开日.png")


def main():
    rows = evaluate_all()
    best_idx = optimum(rows)
    best = rows[best_idx]
    choices, by_vsl = sensitivity()
    p10 = percentile_date(choices, 0.10)
    p50 = percentile_date(choices, 0.50)
    p90 = percentile_date(choices, 0.90)
    chart(rows, best_idx, p10, p90)

    with (OUT / "防疫模型_2020后放开时点扫描.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["full_release_date", "projected_deaths", "death_loss_trillion_rmb",
                         "continued_control_loss_trillion_rmb", "governance_stress_loss_trillion_rmb",
                         "release_wave_loss_trillion_rmb",
                         "total_loss_trillion_rmb", "peak_critical_care_over_capacity", "feasible"])
        for r in rows:
            writer.writerow([r["date"], round(r["deaths"]), r["health_loss"]/TRILLION,
                             r["control_loss"]/TRILLION, r["governance_loss"]/TRILLION,
                             r["release_wave_loss"]/TRILLION,
                             r["total_loss"]/TRILLION, r["peak_capacity_ratio"],
                             r["peak_capacity_ratio"] * 1.20 <= 1.15])

    result = {
        "model": "time-varying epidemic counterfactual + monetized mortality + cumulative control cost",
        "scan_range": [SCAN_START.isoformat(), SCAN_END.isoformat()],
        "default_vsl_rmb_per_death": 5_000_000,
        "default_spatial_clustering_multiplier": 1.20,
        "default_optimal_full_release_date": best["date"].isoformat(),
        "recommended_preparation_and_gradual_relaxation_start": (best["date"]-timedelta(days=45)).isoformat(),
        "default_optimum": {
            "projected_deaths": round(best["deaths"]),
            "death_loss_trillion_rmb": best["health_loss"]/TRILLION,
            "continued_control_loss_trillion_rmb": best["control_loss"]/TRILLION,
            "governance_stress_loss_trillion_rmb": best["governance_loss"]/TRILLION,
            "release_wave_loss_trillion_rmb": best["release_wave_loss"]/TRILLION,
            "total_loss_trillion_rmb": best["total_loss"]/TRILLION,
            "peak_critical_care_over_capacity": best["peak_capacity_ratio"],
            "cluster_adjusted_peak_over_capacity": best["peak_capacity_ratio"] * 1.20,
        },
        "all_scenarios": {"p10": p10.isoformat(), "median": p50.isoformat(),
                          "p90": p90.isoformat(), "scenario_count": len(choices)},
        "by_value_of_statistical_life": [
            {"vsl_rmb": x["vsl_rmb"], "p10": x["p10"].isoformat(),
             "median": x["median"].isoformat(), "p90": x["p90"].isoformat()}
            for x in by_vsl
        ],
        "warning": "Counterfactual deaths, VSL and containment-attributable economic costs are not uniquely identifiable. The interval is more defensible than the point estimate.",
    }
    (OUT / "防疫模型_经济死亡权衡最优放开日.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
