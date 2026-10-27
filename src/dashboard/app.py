import os
import sys
import time
import yaml
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Path resolution
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, BASE_DIR)

from src.storage.run_store import RunStore
from src.storage.schema import STANDIN_JOB_PREFIX
from src.comparison.diff_engine import DiffEngine
from src.reporting.report_generator import ReportGenerator
from src.dashboard.jobs import clear_job, get_job, is_running, start_job
from src.gateway.env_loader import load_env_file
from src.gateway.model_client import AUTO_PROVIDER_PRIORITY

load_env_file()  # reads .env without overwriting variables already set in the shell

st.set_page_config(
    page_title="NeMo Evaluator - LLM Benchmarking Platform",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
      div[data-testid="stMetric"] {
          background: rgba(128, 128, 128, 0.08);
          border: 1px solid rgba(128, 128, 128, 0.18);
          border-radius: 12px;
          padding: 14px 18px;
      }
      .stTabs [data-baseweb="tab-list"] { gap: 6px; }
      .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 16px; }
      .mode-live { color: #0f9d58; font-weight: 600; }
      .mode-simulated { color: #b26a00; font-weight: 600; }
      .mode-unknown { color: #777; font-weight: 600; }
      .small-note { color: rgba(128, 128, 128, 0.95); font-size: 0.86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

PROVIDER_KEY = {
    "nvidia_nim": "NVIDIA_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
}
PROVIDER_LABEL = {
    "mock": "Simulator (mock)",
    "auto": "Auto (live route)",
    "nvidia_nim": "NVIDIA NIM",
    "groq": "Groq",
    "openrouter": "OpenRouter",
    "openai": "OpenAI",
    "ollama": "Ollama (local)",
}
PREFERRED_STANDINS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "meta/llama-3.2-11b-vision-instruct",
    "google/gemma-3-12b-it",
]
PREFERRED_DOMAINS = [
    "General Knowledge",
    "Mathematical Reasoning",
    "Logical Reasoning",
    "Code Generation",
    "Question Answering",
    "Compliance & Regulation",
    "Banking & Regulation",
    "Computer Vision",
    "Custom Evaluation",
]
BENCH_WARNINGS = {
    "gpqa": "Every shipped GPQA answer is 'A', so a model that always picks A scores 100%. Results are inflated by position bias.",
    "truthfulqa": "Every shipped TruthfulQA answer is 'A', so results are inflated by position bias.",
    "image_processing": "This dataset still holds 15 placeholder rows without sources. Treat its scores as placeholders.",
    "kyc_india": "100 rows, but the release check currently fails (fewer than 20% of rows marked human_checked).",
    "rbi_regulations": "100 rows, but the release check currently fails (fewer than 20% of rows marked human_checked).",
    "custom_byob": "Only 2 sample rows are present.",
}


# ---------------------------------------------------------------- config / helpers
def load_yaml(rel_path):
    path = os.path.join(BASE_DIR, rel_path)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return {}


def model_is_standin(config):
    return str(config.get("name", "")).lower().startswith("stand-in")


def job_is_standin(job_name):
    return str(job_name or "").lower().startswith(STANDIN_JOB_PREFIX)


def mode_text(mode):
    mode = (mode or "unknown").lower()
    return {"live": "LIVE", "simulated": "SIMULATED"}.get(mode, "UNKNOWN")


def mode_badge(mode):
    mode = (mode or "unknown").lower()
    css = {"live": "mode-live", "simulated": "mode-simulated"}.get(mode, "mode-unknown")
    return f'<span class="{css}">{mode_text(mode)}</span>'


def auto_route_plan(config):
    """The route an auto model will use now: (provider, route dict, key variable) or (None, None, None)."""
    routes = config.get("routes") or {}
    for provider in AUTO_PROVIDER_PRIORITY:
        route = routes.get(provider)
        if not route:
            continue
        if isinstance(route, str):
            route = {"id": route}
        key_name = PROVIDER_KEY.get(provider)
        if key_name and os.environ.get(key_name):
            return provider, route, key_name
    return None, None, None


def provider_ready(config):
    provider = config.get("provider", "mock")
    if provider == "mock":
        return "Simulated - no API call is made"
    if provider == "ollama":
        return "Local Ollama - no vendor key needed"
    if provider == "auto":
        route_provider, route, key_name = auto_route_plan(config)
        if route:
            return f"Live auto-route ready: {PROVIDER_LABEL.get(route_provider, route_provider)} -> {route['id']} ({key_name} loaded)"
        routes = config.get("routes") or {}
        wanted = ", ".join(
            f"{PROVIDER_LABEL.get(p, p)} ({PROVIDER_KEY.get(p, '?')})"
            for p in AUTO_PROVIDER_PRIORITY
            if p in routes
        )
        return f"Live auto-route NOT ready: set one of {wanted} (the run will fail)"
    key_name = config.get("api_key_env") or PROVIDER_KEY.get(provider)
    if not key_name:
        return f"Provider '{provider}' - no key mapping known"
    if os.environ.get(key_name):
        return f"Live - {key_name} loaded from the environment"
    return f"Live - MISSING {key_name} (the run will fail)"


def auto_job_name(model_id, standin):
    short = model_id.split("/")[-1].replace("-", "_").replace(".", "_")[:32]
    prefix = "standin_dashboard_" if standin else "dashboard_"
    return (prefix + short).lower()


@st.cache_data(show_spinner=False)
def dataset_size(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def run_label(row):
    standin = " [stand-in]" if job_is_standin(row.get("job_name")) else ""
    substitute = " [SUBSTITUTE]" if row.get("model_substituted") else ""
    return (
        f"{row['model_name']} | {mode_text(row.get('run_mode'))} | {row.get('job_name', '')}{standin}{substitute}"
        f" | {row['overall_accuracy'] * 100:.1f}% | {row['run_id']}"
    )


def combined_reliability(diff_row):
    parts = []
    if diff_row.get("unreliable_a"):
        parts.append(f"A {diff_row['unreliable_fraction_a']:.0%}")
    if diff_row.get("unreliable_b"):
        parts.append(f"B {diff_row['unreliable_fraction_b']:.0%}")
    return "UNRELIABLE (" + ", ".join(parts) + ")" if parts else "ok"


def job_elapsed(job):
    end = job.get("finished_at") or time.time()
    return max(0, int(end - (job.get("started_at") or end)))


def render_job_result(job, key):
    """The static panel shown once a background run has stopped polling."""
    artifact = job.get("artifact")
    if job.get("status") == "error":
        st.error(f"Run could not finish: {job.get('error')}")
        if job.get("checkpoints"):
            st.warning("Finished benchmarks were saved as INCOMPLETE checkpoint records; they cannot be compared or gated.")
            for checkpoint in job["checkpoints"][-3:]:
                st.caption(f"`{checkpoint['run_id']}` - {', '.join(checkpoint['benchmarks'])} - {checkpoint['path']}")
        else:
            st.caption("No checkpoint was saved. Check the model/provider configuration and try again.")
        if st.button("Dismiss status", key=f"dismiss_{key}"):
            clear_job(key)
            st.session_state.active_job = None
            st.rerun(scope="app")
        return
    if artifact is None:
        st.warning("The run ended without an artifact.")
        if st.button("Dismiss status", key=f"dismiss_{key}"):
            clear_job(key)
            st.session_state.active_job = None
            st.rerun(scope="app")
        return

    if job["status"] == "completed":
        st.success(
            f"{artifact.run_mode.upper()} run `{artifact.run_id}` finished - "
            f"{artifact.overall_accuracy * 100:.1f}% overall accuracy. Saved to `{job.get('saved_path')}`."
        )
    elif job["status"] == "failed":
        st.error(f"Run FAILED: {artifact.failure_reason} Saved to `{job.get('saved_path')}`.")
    elif job["status"] == "incomplete":
        st.warning(f"Run INCOMPLETE: {artifact.failure_reason} Saved to `{job.get('saved_path')}`.")
    else:
        st.info(f"Run status: {job['status']}")

    st.caption(
        f"Requested model: `{artifact.model_id}` | Served as: `{artifact.served_model_id or artifact.model_id}` | "
        f"Provider: {artifact.provider} | Elapsed: {job_elapsed(job)}s"
    )
    if artifact.model_substituted:
        st.warning(
            f"SUBSTITUTE ROUTE: requested `{artifact.model_id}` but the provider served `{artifact.served_model_id}`. "
            f"{artifact.route_note or ''} Results are for the served model, not the exact requested variant."
        )
    if artifact.is_standin:
        st.warning("STAND-IN RUN: pipeline validation only, not a result for the real models.")

    if artifact.benchmark_summaries:
        rows = [
            {
                "Benchmark": summary.benchmark_name,
                "Domain": summary.domain,
                "Accuracy": summary.accuracy,
                "Items": summary.total_samples,
                "API errors": summary.api_error_count,
                "Truncated": summary.truncated_count,
                "No answer": summary.no_answer_count,
                "Reliability": "UNRELIABLE" if summary.unreliable else "ok",
            }
            for summary in artifact.benchmark_summaries.values()
        ]
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
            column_config={
                "Accuracy": st.column_config.ProgressColumn(
                    "Accuracy", min_value=0.0, max_value=1.0, format="percent"
                )
            },
        )
    if artifact.unreliable:
        st.warning(
            f"UNRELIABLE RUN: {artifact.unreliable_fraction:.0%} of items were api_error, truncated or "
            "no_answer_found. This accuracy is not a fair measure of the model."
        )
    if artifact.unreliable_benchmarks:
        st.warning(f"UNRELIABLE BENCHMARK(S): {', '.join(artifact.unreliable_benchmarks)}")

    if st.button("Dismiss status", key=f"dismiss_{key}"):
        clear_job(key)
        st.session_state.active_job = None
        st.rerun(scope="app")


@st.fragment(run_every=2)
def render_running_job(key):
    job = get_job(key)
    if not is_running(job):
        st.rerun(scope="app")
        return
    done = len(job.get("finished_benchmarks") or [])
    total = max(1, len(job.get("benchmark_ids") or []))
    st.info(
        f"**Run in progress:** {job['model_name']} | {job_elapsed(job)}s elapsed | "
        f"{done}/{total} benchmarks finished"
    )
    st.progress(min(1.0, done / total))
    if job.get("finished_benchmarks"):
        st.caption("Finished: " + ", ".join(job["finished_benchmarks"]))
    if job.get("run_id"):
        st.caption(
            f"Run id `{job['run_id']}` - the partial record is INCOMPLETE until the run finishes."
        )
    st.caption("Polling every 2 seconds. You can switch tabs; the run continues in the background.")


models_config = load_yaml("configs/models_config.yaml").get("models", {})
benchmarks_config = load_yaml("configs/benchmarks_config.yaml").get("benchmarks", {})
store = RunStore(os.path.join(BASE_DIR, "runs"))
if "active_job" not in st.session_state:
    st.session_state.active_job = None

standin_models = {k: v for k, v in models_config.items() if model_is_standin(v)}
other_models = {k: v for k, v in models_config.items() if not model_is_standin(v)}
standin_models = dict(
    sorted(standin_models.items(), key=lambda item: (PREFERRED_STANDINS.index(item[0]) if item[0] in PREFERRED_STANDINS else 99, item[0]))
)


# ---------------------------------------------------------------- header / sidebar
st.title("NeMo Evaluator")
st.markdown(
    "**Benchmarking and comparing large language models** | Vishwakarma Institute of Technology, Pune "
    "& Tata Consultancy Services (TCS)"
)
st.caption(
    "LIVE stand-ins make real API calls but are pipeline-validation models, not the models under study. "
    "SIMULATED runs come from the built-in mock provider and say nothing about a real model."
)

with st.sidebar:
    st.header("Status")
    for key in ("NVIDIA_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        st.write(("Loaded  " if os.environ.get(key) else "Missing  ") + f"`{key}`")
    st.caption("Only whether a key exists is shown. Values are never read into the page.")

    st.divider()
    st.subheader("Data files")
    for bench_id in benchmarks_config:
        size = dataset_size(os.path.join(BASE_DIR, "datasets", bench_id, "sample.jsonl"))
        st.write(f"`{bench_id}` - {size} rows")
    st.caption("The three new datasets are not all release-ready. Check the warnings in the Launch tab.")

    st.divider()
    st.subheader("Reminder")
    st.caption(
        "Stand-in runs are labelled by a job name that starts with `standin_`. They are real live results "
        "for the stand-in model only and must never be reported as results for the required models."
    )


tab_leaderboard, tab_radar, tab_diff, tab_run, tab_reports = st.tabs(
    ["Leaderboard", "Radar & domains", "Model diff & flips", "Launch evaluation", "Reports"]
)


# ---------------------------------------------------------------- tab 1: leaderboard
with tab_leaderboard:
    st.subheader("Cross-model leaderboard")
    all_runs = store.list_runs()

    if not all_runs:
        st.info("No completed evaluation runs yet. Use the Launch evaluation tab to create one.")
    else:
        live_count = sum(1 for run in all_runs if run.get("run_mode") == "live")
        simulated_count = sum(1 for run in all_runs if run.get("run_mode") == "simulated")
        standin_count = sum(1 for run in all_runs if job_is_standin(run.get("job_name")))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Completed runs", len(all_runs))
        c2.metric("Live runs", live_count)
        c3.metric("Simulated runs", simulated_count)
        c4.metric("Stand-in runs", standin_count)

        f1, f2 = st.columns([2, 1])
        with f1:
            selected_modes = st.multiselect(
                "Show run modes",
                ["live", "simulated", "unknown"],
                default=["live", "simulated", "unknown"],
            )
        with f2:
            include_standin = st.toggle("Include stand-in runs", value=True)

        if include_standin and standin_count:
            st.warning(
                "Stand-in runs are included below. They are LIVE pipeline-validation runs for stand-in "
                "models only - not results for the real models under study."
            )
        if "simulated" in selected_modes and simulated_count:
            st.info("Simulated rows come from the built-in mock provider, not from a real model.")
        substitute_count = sum(1 for run in all_runs if run.get("model_substituted"))
        if substitute_count:
            st.warning(
                f"{substitute_count} run(s) used a substitute route: the served model was not the exact requested "
                "variant. The 'Served / route' column marks them. Do not report them as results for the requested model."
            )

        leaderboard = store.get_leaderboard()
        filtered = [
            row
            for row in leaderboard
            if (row.get("run_mode") or "unknown") in selected_modes
            and (include_standin or not job_is_standin(row.get("latest_job_name")))
        ]

        if not filtered:
            st.info("No leaderboard rows match the current filters.")
        else:
            df_lb = pd.DataFrame(filtered)
            df_lb["mode"] = df_lb["run_mode"].apply(mode_text)
            df_lb["job"] = df_lb["latest_job_name"].fillna("")
            df_lb["served"] = df_lb.apply(
                lambda row: f"{row['served_model_id']}{' (SUBSTITUTE)' if row['model_substituted'] else ''}",
                axis=1,
            )
            df_lb["ci"] = df_lb.apply(
                lambda row: f"[{row['ci_lower'] * 100:.1f}%, {row['ci_upper'] * 100:.1f}%]",
                axis=1,
            )

            st.dataframe(
                df_lb[
                    [
                        "model_name",
                        "provider",
                        "served",
                        "mode",
                        "job",
                        "best_accuracy",
                        "ci",
                        "avg_latency",
                        "total_tokens",
                        "total_cost",
                        "total_runs",
                        "latest_run_id",
                    ]
                ],
                width="stretch",
                hide_index=True,
                column_config={
                    "model_name": "Model requested",
                    "provider": "Provider",
                    "served": "Served / route",
                    "mode": "Mode",
                    "job": "Latest job",
                    "best_accuracy": st.column_config.ProgressColumn(
                        "Best accuracy", min_value=0.0, max_value=1.0, format="percent"
                    ),
                    "ci": "95% CI",
                    "avg_latency": st.column_config.NumberColumn("Mean latency (s)", format="%.2f"),
                    "total_tokens": st.column_config.NumberColumn("Total tokens", format="%d"),
                    "total_cost": st.column_config.NumberColumn("Est. cost (USD)", format="$%.4f"),
                    "total_runs": "Runs",
                    "latest_run_id": "Latest run ID",
                },
            )

            chart_col_1, chart_col_2 = st.columns(2)
            with chart_col_1:
                fig_bar = px.bar(
                    df_lb,
                    x="model_name",
                    y="best_accuracy",
                    color="mode",
                    title="Best accuracy by model and mode",
                    labels={"best_accuracy": "Best accuracy", "model_name": "Model", "mode": "Mode"},
                    text=df_lb["best_accuracy"].apply(lambda value: f"{value * 100:.1f}%"),
                )
                fig_bar.update_traces(textposition="outside")
                fig_bar.update_layout(
                    yaxis_range=[0, 1.08],
                    legend_orientation="h",
                    legend_y=-0.25,
                    margin=dict(t=50, b=10),
                )
                st.plotly_chart(fig_bar, width="stretch")

            with chart_col_2:
                fig_scatter = px.scatter(
                    df_lb,
                    x="avg_latency",
                    y="best_accuracy",
                    size="total_tokens",
                    color="mode",
                    hover_name="model_name",
                    title="Accuracy vs. latency (bubble = tokens)",
                    labels={"avg_latency": "Mean latency (s)", "best_accuracy": "Best accuracy", "mode": "Mode"},
                )
                fig_scatter.update_layout(
                    yaxis_range=[0, 1.08],
                    legend_orientation="h",
                    legend_y=-0.25,
                    margin=dict(t=50, b=10),
                )
                st.plotly_chart(fig_scatter, width="stretch")


# ---------------------------------------------------------------- tab 2: radar
with tab_radar:
    st.subheader("Domain analysis")
    runs = store.list_runs()

    if not runs:
        st.info("No completed runs yet.")
    else:
        run_options = {run_label(row): row["run_id"] for row in runs}
        default_labels = list(run_options.keys())[: min(4, len(run_options))]
        selected_run_labels = st.multiselect(
            "Runs to show",
            options=list(run_options.keys()),
            default=default_labels,
        )

        if selected_run_labels:
            artifacts = []
            for label in selected_run_labels:
                artifact = store.get_run(run_options[label])
                if artifact:
                    artifacts.append(artifact)

            present_domains = {summary.domain for artifact in artifacts for summary in artifact.benchmark_summaries.values()}
            categories = [domain for domain in PREFERRED_DOMAINS if domain in present_domains]
            categories += sorted(present_domains - set(categories))

            if len(categories) < 2:
                st.info("At least two benchmark domains are needed for a radar chart.")
            else:
                fig_radar = go.Figure()
                for artifact in artifacts:
                    values = []
                    for category in categories:
                        scores = [
                            summary.accuracy
                            for summary in artifact.benchmark_summaries.values()
                            if summary.domain == category
                        ]
                        values.append(sum(scores) / len(scores) if scores else 0.0)
                    values.append(values[0])

                    suffix = " [stand-in]" if artifact.is_standin else ""
                    fig_radar.add_trace(
                        go.Scatterpolar(
                            r=values,
                            theta=categories + [categories[0]],
                            fill="toself",
                            name=f"{artifact.model_name} ({mode_text(artifact.run_mode)}){suffix}",
                        )
                    )

                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1.0])),
                    showlegend=True,
                    title="Model competency across benchmark domains",
                    legend_orientation="h",
                    legend_y=-0.15,
                    margin=dict(t=60, b=10),
                )
                st.plotly_chart(fig_radar, width="stretch")
                st.caption("Domains with no selected benchmark show 0. New datasets appear only when they are in the run.")

            st.markdown("#### Benchmark detail")
            detail_rows = []
            for artifact in artifacts:
                for summary in artifact.benchmark_summaries.values():
                    detail_rows.append(
                        {
                            "Model": artifact.model_name,
                            "Mode": mode_text(artifact.run_mode),
                            "Job": artifact.job_name,
                            "Benchmark": summary.benchmark_name,
                            "Domain": summary.domain,
                            "Accuracy": summary.accuracy,
                            "Items": summary.total_samples,
                            "API errors": summary.api_error_count,
                            "Truncated": summary.truncated_count,
                            "No answer": summary.no_answer_count,
                            "Reliability": "UNRELIABLE" if summary.unreliable else "ok",
                        }
                    )
            if detail_rows:
                st.dataframe(
                    pd.DataFrame(detail_rows),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Accuracy": st.column_config.ProgressColumn(
                            "Accuracy", min_value=0.0, max_value=1.0, format="percent"
                        )
                    },
                )


# ---------------------------------------------------------------- tab 3: diff
with tab_diff:
    st.subheader("Item-level run comparison")
    runs = store.list_runs()

    if len(runs) < 2:
        st.warning("At least two completed runs are required for a comparison.")
    else:
        run_options = {run_label(row): row["run_id"] for row in runs}
        labels = list(run_options.keys())
        c1, c2 = st.columns(2)
        with c1:
            run_a_label = st.selectbox("Baseline - Run A", labels, index=0)
        with c2:
            run_b_label = st.selectbox("Candidate - Run B", labels, index=min(1, len(labels) - 1))

        if run_a_label == run_b_label:
            st.info("Select two different runs to compare.")
        else:
            run_a = store.get_run(run_options[run_a_label])
            run_b = store.get_run(run_options[run_b_label])
            diff = DiffEngine.compare_runs(run_a, run_b)

            if run_a.is_standin or run_b.is_standin:
                st.warning(
                    "STAND-IN RUN: at least one side is a stand-in model. This is pipeline validation only, "
                    "not a result for the real models."
                )
            if run_a.model_substituted or run_b.model_substituted:
                st.warning(
                    "SUBSTITUTE ROUTE: at least one side was served by a model that is not the exact requested "
                    "variant. Check the served id and route note before treating the comparison as a result for "
                    "the requested models."
                )
            if run_a.run_mode != run_b.run_mode:
                st.warning("The two runs have different modes (live vs simulated); the comparison is not meaningful.")
            if "simulated" in (run_a.run_mode, run_b.run_mode):
                st.info("At least one run is simulated (mock provider), so it does not measure a real model.")
            if run_a.unreliable or run_b.unreliable:
                st.warning(
                    f"Unreliable run: A = {run_a.unreliable_fraction:.0%}, B = {run_b.unreliable_fraction:.0%} "
                    "of items were api_error / truncated / no_answer_found."
                )
            if run_a.scorer_version != run_b.scorer_version:
                st.warning("The runs were scored with different scorer versions, so scores are not directly comparable.")
            if not diff["total_aligned_items"]:
                st.error("The two runs have no question IDs in common, so no paired comparison is possible.")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy delta (B - A)", f"{diff['delta_accuracy'] * 100:+.1f}%")
            m2.metric("Latency delta", f"{diff['delta_latency']:+.2f}s", delta_color="inverse")
            m3.metric("B-only correct (flips)", diff["b_only_correct_count"])
            m4.metric("A-only correct (flips)", diff["a_only_correct_count"])

            st.markdown(
                f"**McNemar paired test:** chi-squared = `{diff['mcnemar']['statistic']}`, "
                f"p-value = `{diff['mcnemar']['p_value']}` - "
                + ("statistically significant at p < 0.05" if diff["mcnemar"]["significant"] else "not significant at p < 0.05")
            )

            st.markdown("#### Benchmark breakdown")
            breakdown = pd.DataFrame(
                [
                    {
                        "Benchmark": data["benchmark_name"],
                        "Domain": data["domain"],
                        f"{run_a.model_name} (A)": data["accuracy_a"],
                        f"{run_b.model_name} (B)": data["accuracy_b"],
                        "Delta": data["delta"] * 100,
                        "Reliability": combined_reliability(data),
                    }
                    for _, data in diff["benchmarks"].items()
                ]
            )
            st.dataframe(
                breakdown,
                width="stretch",
                hide_index=True,
                column_config={
                    f"{run_a.model_name} (A)": st.column_config.ProgressColumn(
                        f"{run_a.model_name} (A)", min_value=0.0, max_value=1.0, format="percent"
                    ),
                    f"{run_b.model_name} (B)": st.column_config.ProgressColumn(
                        f"{run_b.model_name} (B)", min_value=0.0, max_value=1.0, format="percent"
                    ),
                    "Delta": st.column_config.NumberColumn("Delta", format="%+.1f%%"),
                },
            )

            st.markdown("#### Flip inspector")
            flip_side = st.radio(
                "Show flips",
                ["Candidate wins (B correct, A wrong)", "Baseline wins (A correct, B wrong)"],
                horizontal=True,
            )
            active_flips = diff["b_wins_flips"] if "Candidate" in flip_side else diff["a_wins_flips"]

            if not active_flips:
                st.info("No flips in this category.")
            else:
                if len(active_flips) > 1:
                    max_flips = st.slider("Flips to show", 1, min(25, len(active_flips)), min(10, len(active_flips)))
                else:
                    max_flips = 1
                for index, flip in enumerate(active_flips[:max_flips]):
                    with st.expander(f"{index + 1}. [{flip['benchmark']}] {flip['question_id']}"):
                        st.markdown(f"**Question:** {flip['prompt']}")
                        st.markdown(f"**Ground truth:** `{flip['ground_truth']}`")
                        left, right = st.columns(2)
                        with left:
                            st.error(f"{run_a.model_name} answered: {flip['extracted_a']}")
                            st.code(flip["response_a"] or "")
                        with right:
                            st.success(f"{run_b.model_name} answered: {flip['extracted_b']}")
                            st.code(flip["response_b"] or "")


# ---------------------------------------------------------------- tab 4: launch
with tab_run:
    st.subheader("Launch an evaluation")
    st.caption(
        "Runs execute in the background, so the page stays responsive while benchmarks run. "
        "Live stand-ins are selected by default. The six real pair models use `provider: auto`: when "
        "OPENROUTER_API_KEY is loaded they call OpenRouter (paid), and the run records which model was served. "
        "A route marked SUBSTITUTE is not the exact requested variant."
    )

    active_key = st.session_state.get("active_job")
    active_job = get_job(active_key)
    running = is_running(active_job)
    if running:
        st.markdown("#### Current run")
        render_running_job(active_key)
    elif active_job:
        st.markdown("#### Last run from this page")
        render_job_result(active_job, active_key)
        st.divider()

    scope = st.radio(
        "Model set",
        ["Live stand-ins", "All configured models"],
        horizontal=True,
        help="Live stand-ins are free-tier models used to validate the pipeline, not the models under study.",
    )
    model_pool = standin_models if scope == "Live stand-ins" else {**standin_models, **other_models}

    if not model_pool:
        st.error("No models are configured.")
    else:
        selected_model = st.selectbox(
            "Model under test",
            list(model_pool.keys()),
            format_func=lambda model_id: f"{model_pool[model_id]['name']} ({PROVIDER_LABEL.get(model_pool[model_id].get('provider'), model_pool[model_id].get('provider'))})",
        )
        model_cfg = model_pool[selected_model]
        provider = model_cfg.get("provider", "mock")
        standin = model_is_standin(model_cfg)

        info_1, info_2, info_3 = st.columns(3)
        info_1.metric("Provider", PROVIDER_LABEL.get(provider, provider))
        info_2.metric("Run mode", "LIVE" if provider != "mock" else "SIMULATED")
        info_3.metric("Model ID", selected_model.split("/")[-1])
        st.markdown(f"{mode_badge('live' if provider != 'mock' else 'simulated')} {provider_ready(model_cfg)}", unsafe_allow_html=True)

        route_provider, route, route_key = (None, None, None)
        if provider == "auto":
            route_provider, route, route_key = auto_route_plan(model_cfg)
            if route:
                st.info(
                    f"Auto route: {PROVIDER_LABEL.get(route_provider, route_provider)} -> `{route['id']}` "
                    f"({route_key} loaded). Requested model: `{selected_model}`."
                )
                if route.get("substitute"):
                    st.warning(
                        f"SUBSTITUTE ROUTE: `{route['id']}` is not the exact requested variant. "
                        f"{route.get('note', '')} Results must be described as results for the served model."
                    )
            else:
                st.error(
                    "No API key is loaded for any route of this auto model. Add one to `.env` "
                    "(for these pair models that is OPENROUTER_API_KEY; OpenRouter is paid) and reload the page."
                )
        if selected_model == "google/gemma-3-12b-it":
            st.error("Known issue: this model returned HTTP 404 on a real call. A run will stop early (FAILED).")
        if standin:
            st.warning(
                "This is a STAND-IN model. Its results are real live results for this model only and are pipeline "
                "validation, never results for the required models."
            )

        selected_benches = st.multiselect(
            "Benchmarks",
            list(benchmarks_config.keys()),
            default=list(benchmarks_config.keys())[:4],
            format_func=lambda bench_id: f"{benchmarks_config[bench_id]['name']} - {benchmarks_config[bench_id]['domain']}",
        )
        for bench_id in selected_benches:
            if bench_id in BENCH_WARNINGS:
                st.warning(BENCH_WARNINGS[bench_id])

        st.markdown("#### Run size & Execution Engine")
        cap_col, solver_col, engine_col = st.columns(3)
        with cap_col:
            use_all = st.checkbox("Use every row in each selected dataset", value=False)
            if use_all:
                sample_limit = 0
                st.caption("0 means every row. This can take a long time on the 100-row datasets.")
            else:
                sample_limit = st.number_input(
                    "Samples per benchmark",
                    min_value=1,
                    max_value=100,
                    value=5,
                    step=1,
                    help="The previous dashboard limit was 15; it now goes up to 100, or use every row above.",
                )
        with solver_col:
            solver = st.selectbox("Prompting strategy", ["simple", "cot", "harbor"], help="harbor is an agentic prompt style, not a separate agent.")
        with engine_col:
            engine_choice = st.selectbox("Evaluation Engine", ["NVIDIA NeMo Evaluator (official)", "Native runner"], index=0, help="Choose between official NVIDIA NeMo Evaluator adapter or native runner.")
            engine = "nemo" if "NeMo" in engine_choice else "native"

        estimated_items = 0
        for bench_id in selected_benches:
            available = dataset_size(os.path.join(BASE_DIR, "datasets", bench_id, "sample.jsonl"))
            estimated_items += available if sample_limit == 0 else min(sample_limit, available)
        st.caption(f"Estimated prompts for this run: {estimated_items}. Runs are sequential.")
        if estimated_items > 100:
            st.warning("This is a large run. Consider a smaller sample cap or fewer benchmarks first.")

        auto_job = auto_job_name(selected_model, standin)
        custom_job = st.checkbox("Customize job name", value=False)
        if custom_job:
            job_name = st.text_input("Job name", value=auto_job)
        else:
            job_name = auto_job
        if standin and not job_is_standin(job_name):
            job_name = STANDIN_JOB_PREFIX + job_name
        st.caption(f"Job name: `{job_name}` - this is what labels the run as a stand-in when applicable.")

        force_mock = st.checkbox("Run in Mock / Simulation mode (Offline, zero API spend)", value=(provider == "auto" and route is None), help="Simulates the model evaluation locally without making live API calls.")
        if force_mock:
            model_cfg = {**model_cfg, "provider": "mock"}
            provider = "mock"
            standin = False

        start = st.button("Start evaluation", type="primary", width="content", disabled=running)
        if start:
            if running:
                st.error("A run is already in progress.")
            elif not selected_benches:
                st.error("Select at least one benchmark.")
            elif not standin and job_is_standin(job_name):
                st.error("A simulated/non-stand-in model cannot use a `standin_` job name. Choose a different job name.")
            elif provider == "auto" and route is None:
                st.error("No API key is loaded for this auto route. Add one to `.env` and reload the page.")
            else:
                if standin:
                    description = (
                        "PIPELINE VALIDATION ONLY. Interactive dashboard run with a stand-in model. "
                        "Not a result for the required models."
                    )
                elif provider == "auto":
                    description = (
                        "Interactive dashboard run using the automatic live route. The served model and any "
                        "substitute note are recorded in the run artifact."
                    )
                elif provider == "mock":
                    description = "Interactive dashboard run using the built-in mock provider (simulated)."
                else:
                    description = f"Interactive dashboard run using live provider {provider}."

                if active_key:
                    clear_job(active_key)
                new_key = start_job(
                    store=store,
                    model_config=model_cfg,
                    benchmark_ids=selected_benches,
                    job_name=job_name,
                    description=description,
                    limit_per_benchmark=sample_limit,
                    solver_type=solver,
                    seed=42,
                    standin=standin,
                    base_dir=BASE_DIR,
                    engine=engine,
                )
                st.session_state.active_job = new_key
                st.rerun(scope="app")


# ---------------------------------------------------------------- tab 5: reports
with tab_reports:
    st.subheader("Automated comparison report")
    runs = store.list_runs()

    if len(runs) < 2:
        st.info("Run at least two evaluations to generate a comparison report.")
    else:
        run_options = {run_label(row): row["run_id"] for row in runs}
        labels = list(run_options.keys())
        c1, c2 = st.columns(2)
        with c1:
            report_a_label = st.selectbox("Report baseline - Run A", labels, index=0)
        with c2:
            report_b_label = st.selectbox("Report candidate - Run B", labels, index=min(1, len(labels) - 1))

        if report_a_label == report_b_label:
            st.info("Select two different runs.")
        else:
            run_a = store.get_run(run_options[report_a_label])
            run_b = store.get_run(run_options[report_b_label])
            comparison = DiffEngine.compare_runs(run_a, run_b)
            markdown_report = ReportGenerator.generate_markdown(comparison)
            html_report = ReportGenerator.generate_html(comparison)

            if run_a.is_standin or run_b.is_standin:
                st.warning("At least one run is a stand-in. The report says so; do not present it as a real-model result.")
            if run_a.model_substituted or run_b.model_substituted:
                st.warning(
                    "At least one run used a substitute route. The report names the served model; do not present "
                    "it as a result for the exact requested model."
                )
            if (run_a.run_mode == "simulated") or (run_b.run_mode == "simulated"):
                st.info("At least one run is simulated. The report says so.")

            download_1, download_2 = st.columns(2)
            with download_1:
                st.download_button(
                    "Download Markdown report",
                    data=markdown_report,
                    file_name="nemo_evaluator_report.md",
                    mime="text/markdown",
                    width="content",
                )
            with download_2:
                st.download_button(
                    "Download HTML report",
                    data=html_report,
                    file_name="nemo_evaluator_report.html",
                    mime="text/html",
                    width="content",
                )

            st.markdown("### Preview")
            st.markdown(markdown_report)
