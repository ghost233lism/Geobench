#!/usr/bin/env python3
"""Sampling result viewer with a richer Chinese web UI."""

from __future__ import annotations

import argparse
import bisect
import json
import mimetypes
import statistics
import urllib.parse
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地启动采样结果可视化页面。")
    parser.add_argument("--run-dir", required=True, help="包含 summary.json / samples.jsonl / image_stats.jsonl 的目录。")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def h(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def pretty_json(value: Any) -> str:
    return h(json.dumps(value, ensure_ascii=False, indent=2))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else 0.0


def classify_issue(row: Dict[str, Any]) -> str | None:
    error = row.get("reward_details", {}).get("error")
    parsed_answer = row.get("parsed_answer")
    if not parsed_answer:
        return "generation_error"
    if error == "judge_json_parse_failed":
        return "judge_error"
    if error and row.get("reward_mode") == "llm_judge":
        return "judge_error"
    if error:
        return "generation_error"
    return None


class SamplingData:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.summary = load_json(run_dir / "summary.json", {})
        self.run_config = load_json(run_dir / "run_config.json", {})
        self.samples = load_jsonl(run_dir / "samples.jsonl")
        self.image_stats = load_jsonl(run_dir / "image_stats.jsonl")
        self.selection = load_jsonl(run_dir / "selection.jsonl")

        self.selected_ids = {row["item_id"] for row in self.selection if row.get("selected")}
        self.samples_by_item: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in self.samples:
            self.samples_by_item[row["item_id"]].append(row)
        self.image_stats_by_id = {row["item_id"]: row for row in self.image_stats}
        for row in self.image_stats:
            row["selected"] = row["item_id"] in self.selected_ids
            item_samples = self.samples_by_item.get(row["item_id"], [])
            generation_error_count = sum(1 for sample in item_samples if classify_issue(sample) == "generation_error")
            judge_error_count = sum(1 for sample in item_samples if classify_issue(sample) == "judge_error")
            row["generation_error_count"] = generation_error_count
            row["judge_error_count"] = judge_error_count
            row["total_error_count"] = generation_error_count + judge_error_count

        self.global_stats = self._compute_global_stats()

    def _compute_global_stats(self) -> Dict[str, Any]:
        reward_values = [float(row.get("reward", 0.0)) for row in self.samples]
        parsed_count = sum(1 for row in self.samples if row.get("parsed_answer"))
        truncated_count = sum(1 for row in self.samples if row.get("reward_details", {}).get("output_truncated"))
        parse_error_count = sum(1 for row in self.samples if row.get("reward_details", {}).get("error"))
        model_counts = Counter(row.get("model_name", "unknown") for row in self.samples)
        model_reward_values: Dict[str, List[float]] = defaultdict(list)
        for row in self.samples:
            model_reward_values[row.get("model_name", "unknown")].append(float(row.get("reward", 0.0)))
        error_counts = Counter(
            row.get("reward_details", {}).get("error")
            for row in self.samples
            if row.get("reward_details", {}).get("error")
        )
        model_totals = Counter(row.get("model_name", "unknown") for row in self.samples)
        model_parsed = Counter(
            row.get("model_name", "unknown")
            for row in self.samples
            if row.get("parsed_answer")
        )
        issue_counts = Counter(classify_issue(row) for row in self.samples if classify_issue(row))
        model_parsed_rate = {
            model: (model_parsed.get(model, 0) / total if total else 0.0)
            for model, total in model_totals.items()
        }
        model_reward_mean = {
            model: mean(values)
            for model, values in model_reward_values.items()
        }
        selected_count = sum(1 for row in self.image_stats if row.get("selected"))
        probability_values = [float(row.get("probability", 0.0)) for row in self.image_stats]
        selected_probability_values = [float(row.get("probability", 0.0)) for row in self.image_stats if row.get("selected")]
        unselected_probability_values = [float(row.get("probability", 0.0)) for row in self.image_stats if not row.get("selected")]
        s_mid_values = [float(row.get("s_mid", 0.0)) for row in self.image_stats]
        s_var_values = [float(row.get("s_var", 0.0)) for row in self.image_stats]
        prob_selection_relation = compute_probability_selection_relation(self.image_stats, bins=10)
        return {
            "num_items": len(self.image_stats),
            "num_samples": len(self.samples),
            "num_selected": selected_count,
            "selected_rate": (selected_count / len(self.image_stats)) if self.image_stats else 0.0,
            "reward_mean": mean(reward_values),
            "reward_median": statistics.median(reward_values) if reward_values else 0.0,
            "reward_min": min(reward_values) if reward_values else 0.0,
            "reward_max": max(reward_values) if reward_values else 0.0,
            "parsed_rate": parsed_count / len(self.samples) if self.samples else 0.0,
            "truncated_rate": truncated_count / len(self.samples) if self.samples else 0.0,
            "parse_error_rate": parse_error_count / len(self.samples) if self.samples else 0.0,
            "probability_mean": mean(probability_values),
            "probability_median": median(probability_values),
            "selected_probability_mean": mean(selected_probability_values),
            "selected_probability_median": median(selected_probability_values),
            "unselected_probability_mean": mean(unselected_probability_values),
            "unselected_probability_median": median(unselected_probability_values),
            "s_mid_mean": mean(s_mid_values),
            "s_var_mean": mean(s_var_values),
            "model_counts": dict(model_counts),
            "model_parsed_rate": model_parsed_rate,
            "model_reward_mean": model_reward_mean,
            "error_counts": dict(error_counts),
            "issue_counts": dict(issue_counts),
            "items_with_generation_error": sum(1 for row in self.image_stats if row.get("generation_error_count", 0) > 0),
            "items_with_judge_error": sum(1 for row in self.image_stats if row.get("judge_error_count", 0) > 0),
            "prob_selection_relation": prob_selection_relation,
        }

    def list_items(
        self,
        query: str = "",
        selected_only: bool = False,
        sort_by: str = "probability",
        issue_filter: str = "all",
    ) -> List[Dict[str, Any]]:
        rows = list(self.image_stats)
        if query:
            q = query.lower()
            rows = [
                row for row in rows
                if q in str(row.get("item_id", "")).lower()
                or q in str(row.get("gt_text", "")).lower()
                or q in str(row.get("relative_path", "")).lower()
            ]
        if selected_only:
            rows = [row for row in rows if row.get("selected")]
        if issue_filter == "generation_error":
            rows = [row for row in rows if row.get("generation_error_count", 0) > 0]
        elif issue_filter == "judge_error":
            rows = [row for row in rows if row.get("judge_error_count", 0) > 0]
        elif issue_filter == "any_error":
            rows = [row for row in rows if row.get("total_error_count", 0) > 0]
        reverse = sort_by not in {"item_id", "gt_text", "relative_path"}
        rows.sort(key=lambda row: row.get(sort_by, 0), reverse=reverse)
        return rows


def render_metric_card(title: str, value: str, sub: str = "") -> str:
    return f"""
    <section class="metric-card">
      <div class="metric-title">{h(title)}</div>
      <div class="metric-value">{h(value)}</div>
      <div class="metric-sub">{h(sub)}</div>
    </section>
    """


def render_bar_chart(title: str, values: Dict[str, int], accent: str) -> str:
    if not values:
        return f"<section class='panel'><h3>{h(title)}</h3><div class='empty'>暂无数据</div></section>"
    max_value = max(values.values()) or 1
    rows = []
    for key, value in sorted(values.items(), key=lambda kv: kv[1], reverse=True):
        width = value / max_value * 100
        rows.append(
            f"""
            <div class="bar-line">
              <div class="bar-label" title="{h(key)}">{h(key)}</div>
              <div class="bar-track"><div class="bar-fill" style="width:{width:.2f}%;background:{accent};"></div></div>
              <div class="bar-number">{value}</div>
            </div>
            """
        )
    return f"<section class='panel'><h3>{h(title)}</h3>{''.join(rows)}</section>"


def render_rate_bar_chart(title: str, values: Dict[str, float], accent: str) -> str:
    if not values:
        return f"<section class='panel'><h3>{h(title)}</h3><div class='empty'>暂无数据</div></section>"
    rows = []
    for key, value in sorted(values.items(), key=lambda kv: kv[1], reverse=True):
        rate = max(0.0, min(1.0, float(value)))
        width = rate * 100
        rows.append(
            f"""
            <div class="bar-line">
              <div class="bar-label" title="{h(key)}">{h(key)}</div>
              <div class="bar-track"><div class="bar-fill" style="width:{width:.2f}%;background:{accent};"></div></div>
              <div class="bar-number">{rate*100:.1f}%</div>
            </div>
            """
        )
    return f"<section class='panel'><h3>{h(title)}</h3>{''.join(rows)}</section>"


def render_float_bar_chart(title: str, values: Dict[str, float], accent: str, precision: int = 4) -> str:
    if not values:
        return f"<section class='panel'><h3>{h(title)}</h3><div class='empty'>暂无数据</div></section>"
    max_value = max(values.values()) or 1.0
    rows = []
    for key, value in sorted(values.items(), key=lambda kv: kv[1], reverse=True):
        numeric_value = max(0.0, float(value))
        width = (numeric_value / max_value * 100) if max_value > 0 else 0.0
        rows.append(
            f"""
            <div class="bar-line">
              <div class="bar-label" title="{h(key)}">{h(key)}</div>
              <div class="bar-track"><div class="bar-fill" style="width:{width:.2f}%;background:{accent};"></div></div>
              <div class="bar-number">{numeric_value:.{precision}f}</div>
            </div>
            """
        )
    return f"<section class='panel'><h3>{h(title)}</h3>{''.join(rows)}</section>"


def render_histogram(title: str, values: List[float], bins: int = 12) -> str:
    if not values:
        return f"<section class='panel'><h3>{h(title)}</h3><div class='empty'>暂无数据</div></section>"
    low, high = min(values), max(values)
    if abs(high - low) < 1e-12:
        buckets = [len(values)]
        labels = [f"{low:.2f}"]
    else:
        step = (high - low) / bins
        buckets = [0 for _ in range(bins)]
        labels = []
        for value in values:
            idx = min(int((value - low) / step), bins - 1)
            buckets[idx] += 1
        for idx in range(bins):
            left = low + idx * step
            right = left + step
            labels.append(f"{left:.2f}-{right:.2f}")
    max_bucket = max(buckets) or 1
    bars = []
    for label, count in zip(labels, buckets):
        height = count / max_bucket * 160
        bars.append(
            f"""
            <div class="hist-col">
              <div class="hist-bar" style="height:{height:.1f}px;"></div>
              <div class="hist-count">{count}</div>
              <div class="hist-label">{h(label)}</div>
            </div>
            """
        )
    return f"<section class='panel'><h3>{h(title)}</h3><div class='hist-wrap'>{''.join(bars)}</div></section>"


def render_scatter(
    items: List[Dict[str, Any]],
    *,
    title: str,
    subtitle: str,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    y_fallback_key: str | None = None,
) -> str:
    if not items:
        return f"<section class='panel'><h3>{h(title)}</h3><div class='empty'>暂无数据</div></section>"
    width = 760
    height = 360
    pad = 36
    xs = [float(row.get(x_key, 0.0)) for row in items]
    ys = [
        float(row.get(y_key, row.get(y_fallback_key, 0.0) if y_fallback_key else 0.0))
        for row in items
    ]
    ps = [float(row.get("probability", 0.0)) for row in items]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    p_max = max(ps) or 1.0

    def sx(value: float) -> float:
        if abs(x_max - x_min) < 1e-12:
            return width / 2
        return pad + (value - x_min) / (x_max - x_min) * (width - pad * 2)

    def sy(value: float) -> float:
        if abs(y_max - y_min) < 1e-12:
            return height / 2
        return height - pad - (value - y_min) / (y_max - y_min) * (height - pad * 2)

    circles = []
    for row, x, y, p in zip(items, xs, ys, ps):
        r = 4 + p / p_max * 10
        color = "#db5c38" if row.get("selected") else "#2d6cdf"
        circles.append(
            f"<circle cx='{sx(x):.2f}' cy='{sy(y):.2f}' r='{r:.2f}' fill='{color}' fill-opacity='0.72'>"
            f"<title>{h(row.get('item_id'))} | {h(x_label)}={x:.4f} | {h(y_label)}={y:.4f} | p={p:.6f}</title></circle>"
        )
    return f"""
    <section class="panel panel-wide">
      <h3>{h(title)}</h3>
      <div class="muted">{h(subtitle)}</div>
      <svg viewBox="0 0 {width} {height}" class="scatter-svg" role="img">
        <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#999" stroke-width="1.2"></line>
        <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#999" stroke-width="1.2"></line>
        {''.join(circles)}
      </svg>
    </section>
    """


def compute_probability_selection_relation(items: List[Dict[str, Any]], bins: int = 10) -> Dict[str, Any]:
    rows: List[tuple[float, bool]] = []
    for row in items:
        rows.append((float(row.get("probability", 0.0)), bool(row.get("selected"))))
    if not rows:
        return {
            "pairwise_auc_like": 0.0,
            "pearson_corr": 0.0,
            "bins": [],
        }

    selected = [p for p, s in rows if s]
    unselected = [p for p, s in rows if not s]

    auc_like = 0.5
    if selected and unselected:
        sorted_unselected = sorted(unselected)
        wins = 0.0
        for p in sorted(selected):
            lt = bisect.bisect_left(sorted_unselected, p)
            le = bisect.bisect_right(sorted_unselected, p)
            wins += lt + 0.5 * (le - lt)
        auc_like = wins / (len(selected) * len(unselected))

    ps = [p for p, _ in rows]
    ys = [1.0 if s else 0.0 for _, s in rows]
    p_mean = mean(ps)
    y_mean = mean(ys)
    cov = mean((p - p_mean) * (y - y_mean) for p, y in zip(ps, ys))
    p_var = mean((p - p_mean) ** 2 for p in ps)
    y_var = mean((y - y_mean) ** 2 for y in ys)
    pearson_corr = cov / ((p_var * y_var) ** 0.5 + 1e-18)

    rows_sorted = sorted(rows, key=lambda x: x[0])
    n = len(rows_sorted)
    bucket_rows = []
    for idx in range(bins):
        left = int(idx * n / bins)
        right = int((idx + 1) * n / bins)
        chunk = rows_sorted[left:right]
        if not chunk:
            continue
        p_vals = [p for p, _ in chunk]
        select_rate = sum(1 for _, s in chunk if s) / len(chunk)
        bucket_rows.append(
            {
                "label": f"D{idx+1}",
                "count": len(chunk),
                "p_min": min(p_vals),
                "p_max": max(p_vals),
                "p_mean": mean(p_vals),
                "select_rate": select_rate,
            }
        )
    return {
        "pairwise_auc_like": auc_like,
        "pearson_corr": pearson_corr,
        "bins": bucket_rows,
    }


def render_probability_bin_retention(relation: Dict[str, Any]) -> str:
    bins = relation.get("bins", [])
    if not bins:
        return "<section class='panel'><h3>概率区间保留率</h3><div class='empty'>暂无数据</div></section>"
    max_rate = max(float(row["select_rate"]) for row in bins) or 1.0
    lines = []
    for row in bins:
        rate = float(row["select_rate"])
        width = (rate / max_rate) * 100 if max_rate > 0 else 0.0
        lines.append(
            f"""
            <div class="bar-line bar-line-wide">
              <div class="bar-label" title="{h(row['label'])}">{h(row['label'])} ({row['count']})</div>
              <div class="bar-track">
                <div class="bar-fill" style="width:{width:.2f}%;background:#1f7a5c;"></div>
              </div>
              <div class="bar-number">{rate*100:.2f}%</div>
              <div class="bar-extra">p∈[{row['p_min']:.2e}, {row['p_max']:.2e}] | μ={row['p_mean']:.2e}</div>
            </div>
            """
        )
    return f"<section class='panel panel-wide'><h3>概率区间 vs 最终保留率（按概率从低到高分桶）</h3>{''.join(lines)}</section>"


def render_table(rows: List[Dict[str, Any]]) -> str:
    body = []
    for row in rows:
        item_id = row["item_id"]
        href = "/item?id=" + urllib.parse.quote(item_id)
        body.append(
            f"""
            <tr>
              <td><a href="{href}">{h(item_id)}</a></td>
              <td>{h(row.get('gt_text', ''))}</td>
              <td>{h(row.get('relative_path', ''))}</td>
              <td>{row.get('num_samples', 0)}</td>
              <td>{row.get('reward_mean', 0):.4f}</td>
              <td>{row.get('reward_std', 0):.4f}</td>
              <td>{row.get('s_mid', 0):.4f}</td>
              <td>{row.get('s_var', 0):.4f}</td>
              <td>{row.get('probability', 0):.6f}</td>
              <td>{row.get('generation_error_count', 0)}</td>
              <td>{row.get('judge_error_count', 0)}</td>
              <td>{'是' if row.get('selected') else '否'}</td>
            </tr>
            """
        )
    return f"""
    <section class="panel panel-wide">
      <h3>样本明细</h3>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>样本 ID</th>
              <th>GT</th>
              <th>相对路径</th>
              <th>采样数</th>
              <th>Reward 均值</th>
              <th>Reward 方差</th>
              <th>S_mid</th>
              <th>S_var</th>
              <th>保留概率</th>
              <th>生成错误数</th>
              <th>Judge 错误数</th>
              <th>已选中</th>
            </tr>
          </thead>
          <tbody>{''.join(body)}</tbody>
        </table>
      </div>
    </section>
    """


def render_home(data: SamplingData, query: str, selected_only: bool, sort_by: str, issue_filter: str) -> str:
    rows = data.list_items(query=query, selected_only=selected_only, sort_by=sort_by, issue_filter=issue_filter)
    reward_values = [float(row.get("reward", 0.0)) for row in data.samples]
    stats = data.global_stats
    relation = stats["prob_selection_relation"]
    metrics = "".join(
        [
            render_metric_card("图片总数", str(stats["num_items"]), "按 image_stats.jsonl 统计"),
            render_metric_card("采样总数", str(stats["num_samples"]), "按 samples.jsonl 统计"),
            render_metric_card("最终保留", str(stats["num_selected"]), "selection.jsonl 中 selected=true"),
            render_metric_card("平均奖励", f"{stats['reward_mean']:.4f}", f"中位数 {stats['reward_median']:.4f}"),
            render_metric_card("答案解析率", f"{stats['parsed_rate']*100:.1f}%", "parsed_answer 非空比例"),
            render_metric_card("Judge 解析失败率", f"{stats['parse_error_rate']*100:.1f}%", "reward_details.error"),
            render_metric_card("截断比例", f"{stats['truncated_rate']*100:.1f}%", "reward_details.output_truncated"),
            render_metric_card("平均 S_mid", f"{stats['s_mid_mean']:.6f}", "中等难度分数均值"),
            render_metric_card("平均 S_var", f"{stats['s_var_mean']:.6f}", "区分度分数均值"),
            render_metric_card("平均保留概率", f"{stats['probability_mean']:.6f}", f"reward range {stats['reward_min']:.2f} ~ {stats['reward_max']:.2f}"),
            render_metric_card("保留率", f"{stats['selected_rate']*100:.2f}%", f"selected={stats['num_selected']} / all={stats['num_items']}"),
            render_metric_card("已保留均值P", f"{stats['selected_probability_mean']:.6f}", f"未保留均值P {stats['unselected_probability_mean']:.6f}"),
            render_metric_card("趋势AUC-like", f"{relation['pairwise_auc_like']:.4f}", "P(p_selected > p_unselected)"),
            render_metric_card("相关系数", f"{relation['pearson_corr']:.4f}", "corr(probability, selected)"),
        ]
    )
    selected_str = "1" if selected_only else "0"
    issue_links = "".join(
        [
            f"<a class='quick-link' href='/?issue_filter=generation_error&sort=generation_error_count'>生成答案出错条目 {stats['items_with_generation_error']}</a>",
            f"<a class='quick-link' href='/?issue_filter=judge_error&sort=judge_error_count'>Judge 出错条目 {stats['items_with_judge_error']}</a>",
            f"<a class='quick-link' href='/?issue_filter=any_error&sort=total_error_count'>全部错误条目 {stats['items_with_generation_error'] + stats['items_with_judge_error']}</a>",
            "<a class='quick-link' href='/'>清除错误筛选</a>",
        ]
    )
    option_html = "".join(
        [
            f"<option value='{h(k)}' {'selected' if sort_by == k else ''}>{h(k)}</option>"
            for k in [
                "probability",
                "reward_mean",
                "reward_std",
                "s_mid",
                "s_var",
                "value",
                "generation_error_count",
                "judge_error_count",
                "total_error_count",
                "item_id",
                "gt_text",
                "relative_path",
            ]
        ]
    )
    issue_filter_html = "".join(
        [
            f"<option value='{value}' {'selected' if issue_filter == value else ''}>{label}</option>"
            for value, label in [
                ("all", "全部"),
                ("generation_error", "只看生成答案出错"),
                ("judge_error", "只看 Judge 出错"),
                ("any_error", "只看任意错误"),
            ]
        ]
    )
    return f"""
    <html>
    <head>
      <meta charset="utf-8">
      <title>采样结果可视化</title>
      <style>
        :root {{
          --bg: #f2efe8;
          --paper: #fffdf8;
          --ink: #1f1d1a;
          --muted: #746f66;
          --line: #d8d0c2;
          --accent: #cc5c3b;
          --accent-2: #2d6cdf;
          --accent-3: #1f7a5c;
          --shadow: 0 14px 40px rgba(39, 30, 11, 0.10);
        }}
        * {{ box-sizing: border-box; }}
        body {{
          margin: 0;
          color: var(--ink);
          background:
            radial-gradient(circle at top left, rgba(204,92,59,0.10), transparent 22%),
            radial-gradient(circle at top right, rgba(45,108,223,0.09), transparent 24%),
            linear-gradient(180deg, #f7f3ec 0%, #f2efe8 100%);
          font-family: Georgia, "Noto Serif CJK SC", "Source Han Serif SC", serif;
        }}
        .shell {{ max-width: 1480px; margin: 0 auto; padding: 28px 24px 40px; }}
        .hero {{
          background: linear-gradient(130deg, rgba(24,28,37,0.96), rgba(57,39,24,0.92));
          color: #f7f1e8;
          border-radius: 28px;
          padding: 28px 28px 22px;
          box-shadow: var(--shadow);
          margin-bottom: 24px;
          position: relative;
          overflow: hidden;
        }}
        .hero::after {{
          content: "";
          position: absolute;
          inset: auto -80px -120px auto;
          width: 280px;
          height: 280px;
          border-radius: 999px;
          background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 65%);
          pointer-events: none;
        }}
        .hero h1 {{ margin: 0 0 8px; font-size: 34px; letter-spacing: 0.02em; }}
        .hero p {{ margin: 0; color: rgba(247,241,232,0.78); max-width: 900px; line-height: 1.6; }}
        .hero-path {{ margin-top: 14px; font-size: 13px; color: rgba(247,241,232,0.65); word-break: break-all; }}
        .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-bottom: 24px; }}
        .metric-card {{
          background: rgba(255,253,248,0.92);
          border: 1px solid rgba(216,208,194,0.9);
          border-radius: 18px;
          padding: 16px 18px;
          box-shadow: var(--shadow);
          min-height: 122px;
        }}
        .metric-title {{ font-size: 13px; color: var(--muted); margin-bottom: 10px; }}
        .metric-value {{ font-size: 28px; font-weight: 700; line-height: 1.1; }}
        .metric-sub {{ margin-top: 8px; font-size: 13px; color: var(--muted); }}
        .toolbar {{
          display: grid;
          grid-template-columns: 2fr 1fr 1fr auto;
          gap: 14px;
          align-items: end;
          margin-bottom: 24px;
        }}
        .control {{
          background: rgba(255,253,248,0.9);
          border: 1px solid var(--line);
          border-radius: 16px;
          padding: 12px 14px;
          box-shadow: var(--shadow);
        }}
        .control label {{ display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; }}
        .control input, .control select {{
          width: 100%;
          border: 0;
          background: transparent;
          font: inherit;
          color: var(--ink);
          outline: none;
        }}
        .submit-btn {{
          padding: 15px 18px;
          border: 0;
          border-radius: 16px;
          background: linear-gradient(135deg, var(--accent), #9e3f24);
          color: #fffaf2;
          font-weight: 700;
          cursor: pointer;
          box-shadow: var(--shadow);
        }}
        .layout {{ display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 18px; margin-bottom: 24px; }}
        .quick-links {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom: 18px; }}
        .quick-link {{
          display:inline-flex;
          align-items:center;
          padding:10px 14px;
          border-radius:999px;
          background: rgba(255,253,248,0.9);
          border: 1px solid var(--line);
          box-shadow: var(--shadow);
          color: var(--ink);
        }}
        .panel {{
          background: rgba(255,253,248,0.92);
          border: 1px solid var(--line);
          border-radius: 22px;
          padding: 18px 18px 16px;
          box-shadow: var(--shadow);
        }}
        .panel-wide {{ margin-bottom: 24px; }}
        .panel h3 {{ margin: 0 0 8px; font-size: 18px; }}
        .muted {{ color: var(--muted); font-size: 13px; line-height: 1.5; }}
        .bar-line {{ display: grid; grid-template-columns: 140px 1fr 50px; gap: 10px; align-items: center; margin: 10px 0; }}
        .bar-line-wide {{ grid-template-columns: 120px 1fr 74px 340px; }}
        .bar-label {{ font-size: 13px; color: var(--ink); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .bar-track {{ height: 12px; border-radius: 999px; background: #ece7dc; overflow: hidden; }}
        .bar-fill {{ height: 100%; border-radius: inherit; }}
        .bar-number {{ text-align: right; font-size: 13px; color: var(--muted); }}
        .bar-extra {{ font-size: 12px; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        .hist-wrap {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(34px, 1fr)); gap: 8px; align-items: end; min-height: 220px; margin-top: 12px; }}
        .hist-col {{ display: flex; flex-direction: column; align-items: center; gap: 6px; }}
        .hist-bar {{ width: 100%; max-width: 30px; border-radius: 10px 10px 6px 6px; background: linear-gradient(180deg, var(--accent-2), #8bb4ff); }}
        .hist-count {{ font-size: 12px; color: var(--muted); }}
        .hist-label {{ font-size: 11px; color: var(--muted); writing-mode: vertical-rl; text-orientation: mixed; min-height: 82px; }}
        .scatter-svg {{ width: 100%; max-width: 100%; height: auto; background: linear-gradient(180deg, rgba(45,108,223,0.03), rgba(204,92,59,0.03)); border-radius: 16px; margin-top: 12px; }}
        .table-wrap {{ overflow: auto; max-height: 700px; border-radius: 16px; border: 1px solid var(--line); }}
        table {{ width: 100%; border-collapse: collapse; background: rgba(255,255,255,0.68); }}
        th, td {{ padding: 12px 14px; border-bottom: 1px solid #ece7dc; font-size: 14px; text-align: left; vertical-align: top; }}
        th {{ position: sticky; top: 0; background: #fbf8f2; z-index: 1; }}
        tbody tr:hover {{ background: rgba(45,108,223,0.05); }}
        a {{ color: var(--accent-2); text-decoration: none; }}
        .empty {{ padding: 24px 0; color: var(--muted); }}
        @media (max-width: 1100px) {{
          .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
          .toolbar {{ grid-template-columns: 1fr 1fr; }}
          .layout {{ grid-template-columns: 1fr; }}
        }}
        @media (max-width: 720px) {{
          .shell {{ padding: 18px 14px 30px; }}
          .metrics {{ grid-template-columns: 1fr; }}
          .toolbar {{ grid-template-columns: 1fr; }}
          .hero h1 {{ font-size: 28px; }}
          .bar-line-wide {{ grid-template-columns: 1fr; gap: 6px; }}
          .bar-number, .bar-extra {{ text-align: left; }}
        }}
      </style>
    </head>
    <body>
      <main class="shell">
        <section class="hero">
          <h1>采样结果可视化面板</h1>
          <p>这个页面把全局统计、模型输出质量、奖励分布和每条样本的细节放到同一个界面里。上半部分先看整体健康度，下半部分再按样本逐条钻取。</p>
          <div class="hero-path">{h(str(data.run_dir))}</div>
        </section>

        <section class="metrics">{metrics}</section>

        <section class="quick-links">{issue_links}</section>

        <form class="toolbar" method="get" action="/">
          <div class="control">
            <label>搜索</label>
            <input name="q" value="{h(query)}" placeholder="item_id / gt_text / relative_path">
          </div>
          <div class="control">
            <label>排序字段</label>
            <select name="sort">{option_html}</select>
          </div>
          <div class="control">
            <label>是否只看已选中</label>
            <select name="selected">
              <option value="0" {'selected' if selected_str == '0' else ''}>全部</option>
              <option value="1" {'selected' if selected_str == '1' else ''}>只看已选中</option>
            </select>
          </div>
          <div class="control">
            <label>错误筛选</label>
            <select name="issue_filter">{issue_filter_html}</select>
          </div>
          <button class="submit-btn" type="submit">更新视图</button>
        </form>

        <section class="layout">
          {render_bar_chart("按模型统计样本数", stats["model_counts"], "#2d6cdf")}
          {render_rate_bar_chart("按模型统计答案解析率", stats["model_parsed_rate"], "#1f7a5c")}
        </section>

        <section class="layout">
          {render_float_bar_chart("按模型统计平均奖励", stats["model_reward_mean"], "#cc5c3b")}
          {render_bar_chart("按错误类型统计（仅错误）", stats["error_counts"], "#cc5c3b")}
        </section>

        <section class="layout">
          {render_bar_chart("按问题来源统计", stats["issue_counts"], "#6d4aff")}
          <section class="panel">
            <h3>按模型统计平均奖励</h3>
            <div class="muted">按 `samples.jsonl` 中每条采样的 `model_name` 聚合 reward 均值，便于直接比较不同模型的平均奖励。</div>
          </section>
        </section>

        <section class="layout">
          {render_histogram("S_mid 分布", [float(row.get("s_mid", 0.0)) for row in data.image_stats])}
          {render_histogram("S_var 分布", [float(row.get("s_var", 0.0)) for row in data.image_stats])}
        </section>

        <section class="layout">
          {render_histogram("保留概率分布", [float(row.get("probability", 0.0)) for row in data.image_stats])}
          {render_histogram("奖励分布直方图", reward_values)}
        </section>

        {render_probability_bin_retention(relation)}

        <section class="panel panel-wide"><h3>说明</h3><div class="muted">`S_mid` 是中等难度分数，`S_var` 是区分度分数，二者相乘得到 `value`，归一化后得到保留概率。可以直接用上面的快捷入口和错误筛选快速定位生成答案出错或 Judge 出错的图片。</div></section>

        {render_scatter(
            rows,
            title="reward mean vs reward std 散点图",
            subtitle="横轴是 reward mean，纵轴是 reward std，点大小是保留概率，橙色表示已入选。",
            x_key="reward_mean",
            y_key="reward_std",
            x_label="reward mean",
            y_label="reward std",
        )}
        {render_scatter(
            rows,
            title="S_mid vs s_std 散点图",
            subtitle="横轴是 S_mid，纵轴是 s_std，点大小是保留概率，橙色表示已入选。",
            x_key="s_mid",
            y_key="s_std",
            x_label="S_mid",
            y_label="s_std",
            y_fallback_key="reward_std",
        )}
        {render_table(rows)}

        <section class="panel panel-wide">
          <h3>运行配置</h3>
          <pre>{pretty_json(data.run_config)}</pre>
        </section>
      </main>
    </body>
    </html>
    """


def render_item(data: SamplingData, item_id: str) -> str:
    stat = data.image_stats_by_id.get(item_id)
    if stat is None:
        return "<html><body><h1>未找到该样本</h1></body></html>"
    samples = sorted(data.samples_by_item.get(item_id, []), key=lambda row: (row.get("model_name", ""), row.get("sample_index", 0)))
    image_url = "/file?path=" + urllib.parse.quote(stat["image_path"])
    sample_sections = []
    for row in samples:
        issue = classify_issue(row)
        issue_label = {
            "generation_error": "生成答案出错",
            "judge_error": "Judge 出错",
            None: "正常",
        }[issue]
        sample_sections.append(
            f"""
            <section class="sample-card" style="border-color:{'#cc5c3b' if issue else 'var(--line)'};">
              <div class="sample-head">
                <div class="sample-title">{h(row.get('model_name', ''))} / 第 {row.get('sample_index', 0)} 个采样</div>
                <div class="sample-reward">reward = {row.get('reward', 0):.4f}</div>
              </div>
              <div class="chip-row">
                <span class="chip">状态: {h(issue_label)}</span>
                <span class="chip">解析答案: {h(row.get('parsed_answer') or 'None')}</span>
                <span class="chip">截断: {'是' if row.get('reward_details', {}).get('output_truncated') else '否'}</span>
                <span class="chip">错误: {h(row.get('reward_details', {}).get('error', 'none'))}</span>
              </div>
              <div class="block-title">reward_details</div>
              <pre>{pretty_json(row.get('reward_details', {}))}</pre>
              <div class="block-title">raw_output</div>
              <pre>{h(row.get('raw_output', ''))}</pre>
            </section>
            """
        )
    return f"""
    <html>
    <head>
      <meta charset="utf-8">
      <title>{h(item_id)}</title>
      <style>
        :root {{
          --bg: #f2efe8;
          --paper: #fffdf8;
          --ink: #1f1d1a;
          --muted: #746f66;
          --line: #d8d0c2;
          --accent: #cc5c3b;
          --accent-2: #2d6cdf;
          --shadow: 0 14px 40px rgba(39, 30, 11, 0.10);
        }}
        * {{ box-sizing: border-box; }}
        body {{
          margin: 0;
          color: var(--ink);
          background:
            radial-gradient(circle at top left, rgba(204,92,59,0.10), transparent 22%),
            radial-gradient(circle at top right, rgba(45,108,223,0.09), transparent 24%),
            linear-gradient(180deg, #f7f3ec 0%, #f2efe8 100%);
          font-family: Georgia, "Noto Serif CJK SC", "Source Han Serif SC", serif;
        }}
        .shell {{ max-width: 1500px; margin: 0 auto; padding: 22px 24px 34px; }}
        .back {{ display: inline-block; margin-bottom: 14px; color: var(--accent-2); text-decoration: none; }}
        .title-card {{
          background: linear-gradient(130deg, rgba(24,28,37,0.96), rgba(57,39,24,0.92));
          color: #f7f1e8;
          border-radius: 24px;
          padding: 22px 24px;
          box-shadow: var(--shadow);
          margin-bottom: 20px;
        }}
        .layout {{ display: grid; grid-template-columns: 420px 1fr; gap: 18px; }}
        .panel, .sample-card {{
          background: rgba(255,253,248,0.92);
          border: 1px solid var(--line);
          border-radius: 22px;
          padding: 18px;
          box-shadow: var(--shadow);
          margin-bottom: 18px;
        }}
        .preview {{
          width: 100%;
          display: block;
          border-radius: 16px;
          border: 1px solid var(--line);
          background: #fff;
        }}
        .meta-line {{ margin: 8px 0; line-height: 1.55; word-break: break-all; }}
        .sample-head {{ display:flex; justify-content:space-between; gap:12px; align-items:center; margin-bottom:12px; }}
        .sample-title {{ font-weight: 700; }}
        .sample-reward {{ color: var(--accent); font-weight: 700; }}
        .chip-row {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px; }}
        .chip {{
          display:inline-flex;
          padding:6px 10px;
          border-radius:999px;
          background:#f2eee5;
          border:1px solid var(--line);
          font-size:12px;
        }}
        .block-title {{ margin: 12px 0 6px; color: var(--muted); font-size: 13px; }}
        pre {{
          margin: 0;
          white-space: pre-wrap;
          word-break: break-word;
          background: #faf7f1;
          border: 1px solid #ece7dc;
          padding: 14px;
          border-radius: 14px;
        }}
        a {{ color: var(--accent-2); text-decoration: none; }}
        @media (max-width: 1100px) {{ .layout {{ grid-template-columns: 1fr; }} }}
      </style>
    </head>
    <body>
      <main class="shell">
        <a class="back" href="/">← 返回总览</a>
        <section class="title-card">
          <h1 style="margin:0 0 8px;">{h(item_id)}</h1>
          <div style="opacity:0.8;">单条样本详情页，适合逐个检查生成结果、judge 输出和保留概率。</div>
        </section>
        <section class="layout">
          <div>
            <section class="panel">
              <img class="preview" src="{image_url}" alt="{h(item_id)}">
            </section>
            <section class="panel">
              <div class="meta-line"><b>GT 文本：</b>{h(stat.get('gt_text', ''))}</div>
              <div class="meta-line"><b>图片路径：</b>{h(stat.get('image_path', ''))}</div>
              <div class="meta-line"><b>相对路径：</b>{h(stat.get('relative_path', ''))}</div>
              <div class="meta-line"><b>是否入选：</b>{'是' if stat.get('selected') else '否'}</div>
              <hr style="border:none;border-top:1px solid var(--line);margin:14px 0;">
              <div class="meta-line"><b>reward_mean：</b>{stat.get('reward_mean', 0):.6f}</div>
              <div class="meta-line"><b>reward_std：</b>{stat.get('reward_std', 0):.6f}</div>
              <div class="meta-line"><b>reward_min / reward_max：</b>{stat.get('reward_min', 0):.4f} / {stat.get('reward_max', 0):.4f}</div>
              <div class="meta-line"><b>S_mid：</b>{stat.get('s_mid', 0):.6f}</div>
              <div class="meta-line"><b>S_var：</b>{stat.get('s_var', 0):.6f}</div>
              <div class="meta-line"><b>value：</b>{stat.get('value', 0):.6f}</div>
              <div class="meta-line"><b>probability：</b>{stat.get('probability', 0):.6f}</div>
              <div class="block-title">sample_rewards</div>
              <pre>{h(stat.get('sample_rewards', []))}</pre>
            </section>
          </div>
          <div>
            {''.join(sample_sections)}
          </div>
        </section>
      </main>
    </body>
    </html>
    """


class AppHandler(BaseHTTPRequestHandler):
    data: SamplingData = None  # type: ignore

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        if parsed.path == "/":
            query = params.get("q", [""])[0]
            selected_only = params.get("selected", ["0"])[0] == "1"
            sort_by = params.get("sort", ["probability"])[0]
            issue_filter = params.get("issue_filter", ["all"])[0]
            self._send_html(render_home(self.data, query, selected_only, sort_by, issue_filter))
            return
        if parsed.path == "/item":
            item_id = params.get("id", [""])[0]
            self._send_html(render_item(self.data, item_id))
            return
        if parsed.path == "/file":
            raw_path = params.get("path", [""])[0]
            self._send_file(Path(raw_path))
            return
        self.send_error(404)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        content_type, _ = mimetypes.guess_type(str(path))
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    data = SamplingData(run_dir)
    AppHandler.data = data
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"可视化页面已启动: http://{args.host}:{args.port}")
    print(f"结果目录: {run_dir}")
    server.serve_forever()


if __name__ == "__main__":
    main()
