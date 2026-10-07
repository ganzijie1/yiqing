import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def seir(days, r0, latent_days, infectious_days, control, e0, i0, immune0=0.0, dt=0.1):
    """Normalized SEIR simulation; control may be a scalar or day->scalar function."""
    n = int(days / dt) + 1
    t = np.arange(n) * dt
    y = np.zeros((n, 4), dtype=float)
    y[0] = [1.0 - immune0 - e0 - i0, e0, i0, immune0]
    sigma = 1.0 / latent_days
    gamma = 1.0 / infectious_days
    beta = r0 * gamma
    incidence = np.zeros(n, dtype=float)

    def rhs(state, time):
        s, e, i, r = state
        u = control(time) if callable(control) else control
        force = beta * max(0.0, 1.0 - u) * s * i
        return np.array([-force, force - sigma * e, sigma * e - gamma * i, gamma * i]), force

    for k in range(n - 1):
        state = y[k]
        time = t[k]
        k1, f1 = rhs(state, time)
        k2, f2 = rhs(state + 0.5 * dt * k1, time + 0.5 * dt)
        k3, f3 = rhs(state + 0.5 * dt * k2, time + 0.5 * dt)
        k4, f4 = rhs(state + dt * k3, time + dt)
        y[k + 1] = np.maximum(0.0, state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0)
        y[k + 1] /= y[k + 1].sum()
        incidence[k + 1] = dt * (f1 + 2*f2 + 2*f3 + f4) / 6.0
    return t, y, incidence


def daily_from_steps(t, values, days):
    out = np.zeros(days)
    for day in range(days):
        mask = (t >= day) & (t < day + 1)
        out[day] = values[mask].sum()
    return out


def census_from_incidence(daily_incidence, probability, delay, mean_los):
    days = len(daily_incidence)
    census = np.zeros(days)
    for infection_day, infections in enumerate(daily_incidence):
        first = infection_day + delay
        for day in range(first, days):
            census[day] += infections * probability * math.exp(-(day - first) / mean_los)
    return census


def calibrate_2022():
    # Targets: peak on day 14 (Dec 21) and 82.4% infected by day 62 (Feb 7).
    def search(r_values, seed_values, current=None):
        best = current
        for effective_r in r_values:
            control = 1.0 - effective_r / 8.2
            for seed in seed_values:
                t, y, inc = seir(
                    63, 8.2, 2.0, 3.0, control,
                    e0=seed * 1.8, i0=seed, immune0=0.03, dt=0.2)
                daily = daily_from_steps(t, inc, 63)
                peak_day = int(np.argmax(daily))
                attack = float(y[-1, 3] - 0.03)
                score = ((peak_day - 14) / 2.0) ** 2 + ((attack - 0.824) / 0.02) ** 2
                if best is None or score < best[0]:
                    best = (score, effective_r, seed, peak_day, attack, t, y, daily)
        return best

    best = search(np.linspace(2.15, 3.8, 32), np.geomspace(0.00025, 0.012, 32))
    _, r_mid, seed_mid, *_ = best
    r_values = np.linspace(max(2.05, r_mid - 0.10), min(4.0, r_mid + 0.10), 35)
    seed_values = np.geomspace(seed_mid / 1.25, seed_mid * 1.25, 35)
    best = search(r_values, seed_values, best)
    return best


def calibrate_2020():
    # Jointly approximate the published post-intervention Rt=0.28 and 96% reduction
    # in cumulative infections by Mar 8 relative to no intervention.
    days = 46
    base = seir(days, 3.54, 3.5, 5.0, 0.0, 0.004, 0.002, 0.0, 0.1)
    base_attack = float(base[1][-1, 3])
    best = None
    for control in np.linspace(0.75, 0.95, 401):
        sim = seir(days, 3.54, 3.5, 5.0, control, 0.004, 0.002, 0.0, 0.1)
        attack = float(sim[1][-1, 3])
        reduction = 1.0 - attack / base_attack
        rt = 3.54 * (1.0 - control)
        score = ((reduction - 0.96) / 0.03) ** 2 + ((rt - 0.28) / 0.08) ** 2
        if best is None or score < best[0]:
            best = (score, control, rt, reduction, base, sim)
    return best


def font(size, bold=False):
    name = "msyhbd.ttc" if bold else "msyh.ttc"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def line_chart(path, title, x, series, y_label, x_labels=None, note=None):
    w, h = 1500, 850
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    left, top, right, bottom = 150, 105, 70, 145
    pw, ph = w-left-right, h-top-bottom
    d.text((left, 35), title, fill="#111827", font=font(36, True))
    ymax = max(max(vals) for _, vals, _ in series) * 1.08
    ymax = max(ymax, 1e-9)
    for j in range(6):
        yy = top + ph * j / 5
        value = ymax * (1 - j/5)
        d.line((left, yy, left+pw, yy), fill="#e5e7eb", width=2)
        d.text((15, yy-14), f"{value:,.2f}", fill="#4b5563", font=font(20))
    d.line((left, top, left, top+ph), fill="#374151", width=3)
    d.line((left, top+ph, left+pw, top+ph), fill="#374151", width=3)
    for label, vals, color in series:
        pts=[]
        for i, val in enumerate(vals):
            px=left+pw*i/max(1,len(vals)-1)
            py=top+ph*(1-val/ymax)
            pts.append((px,py))
        d.line(pts, fill=color, width=5)
    if x_labels:
        for idx,label in x_labels:
            px=left+pw*idx/max(1,len(x)-1)
            d.line((px,top+ph,px,top+ph+8),fill="#374151",width=2)
            d.text((px-35,top+ph+15),label,fill="#374151",font=font(19))
    d.text((15, top-42), y_label, fill="#374151", font=font(22))
    legend_x=left
    for label, _, color in series:
        d.line((legend_x, h-70, legend_x+42, h-70),fill=color,width=6)
        d.text((legend_x+52,h-85),label,fill="#111827",font=font(21))
        legend_x += 330
    if note:
        d.text((left, h-118), note, fill="#6b7280", font=font(17))
    img.save(path)


def main():
    c20 = calibrate_2020()
    _, control20, rt20, reduction20, base20, strict20 = c20
    c22 = calibrate_2022()
    _, re22, seed22, peak_day22, attack22, t22, y22, daily22 = c22

    population22 = 1.412e9
    # Calibrate probabilities so model census peaks match official Jan 5 values.
    hosp_unit = census_from_incidence(daily22 * population22, 1.0, 7, 8)
    severe_unit = census_from_incidence(daily22 * population22, 1.0, 11, 12)
    p_hosp = 1.625e6 / hosp_unit.max()
    p_severe = 128000.0 / severe_unit.max()
    hosp22 = hosp_unit * p_hosp
    severe22 = severe_unit * p_severe

    # Counterfactual severity/capacity assumptions are sensitivity scenarios.
    severe_early_open = severe22 * 1.50
    severe_pre_expansion_capacity = 1.412e9 * 4.0 / 100000.0
    severe_dec_capacity = 181000.0

    daily20_base = daily_from_steps(base20[0], base20[2], 46)
    daily20_strict = daily_from_steps(strict20[0], strict20[2], 46)
    scale20 = 11.08e6 / 1000.0
    line_chart(
        OUT / "防疫模型_2020干预对比.png",
        "2020 Wuhan-style transmission: intervention vs no intervention",
        np.arange(46),
        [
            ("No intervention", daily20_base * scale20, "#b91c1c"),
            ("Calibrated strict control", daily20_strict * scale20, "#166534"),
        ],
        "Daily infections per 1,000 population",
        [(0,"Jan 23"),(15,"Feb 7"),(31,"Feb 23"),(45,"Mar 8")],
        "Illustrative SEIR reconstruction; calibrated to published Rt and infection-reduction benchmarks.")

    line_chart(
        OUT / "防疫模型_2022放开波次.png",
        "Late-2022 transition: calibrated national epidemic wave",
        np.arange(63),
        [
            ("New infections (million/day)", daily22*population22/1e6, "#7c3aed"),
            ("Hospital census (million)", hosp22/1e6, "#0369a1"),
            ("Severe census (million)", severe22/1e6, "#be123c"),
        ],
        "Millions",
        [(0,"Dec 7"),(14,"Dec 21"),(29,"Jan 5"),(62,"Feb 7")],
        "Infection timing/attack rate and hospital peaks calibrated to China CDC/NHC benchmarks.")

    line_chart(
        OUT / "防疫模型_医疗容量敏感性.png",
        "Severe-case census and capacity sensitivity",
        np.arange(63),
        [
            ("Observed-style Dec transition", severe22/1000, "#0369a1"),
            ("Early-opening sensitivity", severe_early_open/1000, "#b91c1c"),
            ("Dec ICU beds", np.full(63,severe_dec_capacity/1000), "#166534"),
            ("Pre-expansion ICU beds", np.full(63,severe_pre_expansion_capacity/1000), "#a16207"),
        ],
        "Thousands",
        [(0,"Dec 7"),(14,"Dec 21"),(29,"Jan 5"),(62,"Feb 7")],
        "Capacity lines are national aggregates; regional and staffing constraints are not represented.")

    summary = {
        "2020": {
            "calibrated_control": control20,
            "effective_reproduction_number": rt20,
            "infection_reduction_by_day_46": reduction20,
            "source_targets": {"Rt": 0.28, "infection_reduction": 0.96},
        },
        "2022": {
            "calibrated_effective_R_after_transition": re22,
            "initial_infectious_seed_fraction": seed22,
            "peak_day_after_dec7": peak_day22,
            "attack_rate_by_feb7": attack22,
            "hospital_probability_calibrated": p_hosp,
            "severe_probability_calibrated": p_severe,
            "hospital_peak": float(hosp22.max()),
            "hospital_peak_day": int(np.argmax(hosp22)),
            "severe_peak": float(severe22.max()),
            "severe_peak_day": int(np.argmax(severe22)),
            "early_open_severe_peak_sensitivity": float(severe_early_open.max()),
            "pre_expansion_capacity": severe_pre_expansion_capacity,
            "dec25_icu_capacity": severe_dec_capacity,
        },
    }
    (OUT / "防疫控制模型_仿真摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (OUT / "防疫控制模型_逐日仿真.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["day_after_2022_12_07", "new_infections", "hospital_census", "severe_census"])
        for day in range(63):
            writer.writerow([day, daily22[day]*population22, hosp22[day], severe22[day]])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
