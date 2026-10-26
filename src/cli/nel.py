import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import os
import sys
import yaml
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.storage.run_store import RunStore
from src.engine.runner import EvaluationRunner
from src.gateway.model_client import ProviderConfigError
from src.gateway.env_loader import load_env_file
from src.storage.schema import STANDIN_JOB_PREFIX, UNRELIABLE_THRESHOLD, UNRELIABLE_BENCHMARK_THRESHOLD
from src.comparison.diff_engine import DiffEngine
from src.reporting.report_generator import ReportGenerator
from src.scoring.statistics import StatisticsEngine

console = Console()

def load_yaml(file_path: str):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@click.group()
def cli():
    """NVIDIA NeMo Evaluator Launcher (nel) CLI - Multi-Model Benchmarking Platform"""
    load_env_file()  # reads .env without overriding variables already set in the shell

@cli.command("list")
def list_cmd():
    """List configured models, benchmarks, and previous runs."""
    models_cfg = load_yaml("configs/models_config.yaml").get("models", {})
    bench_cfg = load_yaml("configs/benchmarks_config.yaml").get("benchmarks", {})
    store = RunStore("runs")
    runs = store.list_runs()

    # Models Table
    m_table = Table(title="Available Models (NVIDIA NIM / Mock / Cloud)", header_style="bold cyan")
    m_table.add_column("Model ID", style="bold")
    m_table.add_column("Display Name")
    m_table.add_column("Provider")
    m_table.add_column("Context Window")
    m_table.add_column("Strengths")

    for m_id, m in models_cfg.items():
        m_table.add_row(m_id, m.get("name", ""), m.get("provider", "mock"), str(m.get("context_window", "")), ", ".join(m.get("strengths", [])))
    console.print(m_table)
    console.print()

    # Benchmarks Table
    b_table = Table(title="Available Benchmarks", header_style="bold green")
    b_table.add_column("Benchmark ID", style="bold")
    b_table.add_column("Name")
    b_table.add_column("Domain")
    b_table.add_column("Scorer")
    b_table.add_column("Description")

    for b_id, b in bench_cfg.items():
        b_table.add_row(b_id, b.get("name", ""), b.get("domain", ""), b.get("scorer", ""), b.get("description", ""))
    console.print(b_table)
    console.print()

    # Runs Table
    r_table = Table(title=f"Saved Evaluation Runs ({len(runs)} runs)", header_style="bold magenta")
    r_table.add_column("Run ID", style="bold")
    r_table.add_column("Model Name")
    r_table.add_column("Served / route")
    r_table.add_column("Mode")
    r_table.add_column("Job")
    r_table.add_column("Accuracy")
    r_table.add_column("Mean Latency")
    r_table.add_column("Tokens")
    r_table.add_column("Timestamp")

    for r in runs[:10]:
        served = r.get("served_model_id") or r.get("model_id") or ""
        route = f"{served} (SUBSTITUTE)" if r.get("model_substituted") else served
        r_table.add_row(
            r["run_id"],
            r["model_name"],
            route,
            "simulated" if r["provider"] == "mock" else "live",
            r["job_name"] or "",
            f"{r['overall_accuracy']*100:.1f}%",
            f"{r['mean_latency']:.2f}s",
            str(r["total_tokens"]),
            r["timestamp"][:19]
        )
    console.print(r_table)

    unusable = store.list_unusable()
    if unusable:
        console.print()
        u_table = Table(title=f"INCOMPLETE or FAILED run records ({len(unusable)}): not in the list above, refused by compare and gate",
                        header_style="bold red")
        u_table.add_column("Run ID", style="bold")
        u_table.add_column("Status")
        u_table.add_column("Model Name")
        u_table.add_column("Benchmarks saved")
        u_table.add_column("Reason")
        for r in unusable:
            u_table.add_row(r["run_id"], r["status"].upper(), r["model_name"],
                            f"{len(r['benchmarks_done'])} of {len(r['planned']) or '?'}", r["reason"][:120])
        console.print(u_table)

@cli.group("eval")
def eval_group():
    """Evaluation operations."""
    pass

@eval_group.command("run")
@click.option("--config", default="configs/eval_config_default.yaml", help="Path to evaluation config YAML")
@click.option("--model", default=None, help="Model ID (overrides config)")
@click.option("--bench", multiple=True, help="Benchmark IDs to evaluate")
@click.option("--limit", default=None, type=int, help="Limit samples per benchmark")
@click.option("--solver", default="simple", type=click.Choice(["simple", "cot", "harbor"]), help="Solver type")
@click.option("--job-name", default=None, help="Custom job name")
def run_cmd(config, model, bench, limit, solver, job_name):
    """Execute evaluation runs across models and benchmarks."""
    cfg = load_yaml(config)
    models_cfg = load_yaml("configs/models_config.yaml").get("models", {})
    
    selected_models = [model] if model else cfg.get("models", ["meta/llama-3.1-70b-instruct"])
    selected_benches = list(bench) if bench else cfg.get("benchmarks", ["mmlu_pro", "gsm8k", "math_500", "gpqa", "humaneval", "truthfulqa"])
    sample_limit = limit if limit is not None else cfg.get("limit_per_benchmark", 10)
    current_job = job_name or cfg.get("job_name", "benchmark_run")

    # A stand-in config must keep its stand-in job name (even if --job-name is passed), otherwise the label is lost.
    if os.path.basename(config).lower().startswith("standin_") and not current_job.lower().startswith(STANDIN_JOB_PREFIX):
        console.print(f"[bold red]Configuration error:[/bold red] {config} is a stand-in config, so the job name must start with "
                      f"'{STANDIN_JOB_PREFIX}' (got '{current_job}'). Stand-in runs must stay labelled as pipeline validation.")
        sys.exit(1)

    console.print(Panel(f"[bold green]Starting NeMo Evaluator Job:[/bold green] {current_job}\n"
                        f"Models: {', '.join(selected_models)}\n"
                        f"Benchmarks: {', '.join(selected_benches)}\n"
                        f"Sample Limit: {sample_limit} | Solver: {solver}",
                        title="NeMo Evaluator Launcher (nel)", border_style="cyan"))

    runner = EvaluationRunner()
    store = RunStore("runs")
    failed_models = []

    for m_id in selected_models:
        m_cfg = models_cfg.get(m_id, {
            "id": m_id,
            "name": m_id.split("/")[-1],
            "provider": "mock",
            "params": {"temperature": 0.1, "max_tokens": 1024, "top_p": 0.95}
        })

        console.print(f"\n[bold yellow]--> Evaluating {m_cfg.get('name', m_id)}...[/bold yellow]")
        checkpoints = []   # (run_id, number of finished benchmarks, path) of the partial saves made so far

        def save_checkpoint(partial):
            checkpoints.append((partial.run_id, len(partial.benchmark_summaries), store.save_checkpoint(partial)))

        try:
            artifact = runner.run_evaluation(
                model_config=m_cfg,
                benchmark_ids=selected_benches,
                job_name=current_job,
                limit_per_benchmark=sample_limit,
                solver_type=solver,
                seed=cfg.get("seed", 42),
                checkpoint=save_checkpoint
            )
        except ProviderConfigError as e:
            console.print(f"[bold red]Configuration error:[/bold red] {e}")
            sys.exit(1)
        except BaseException as e:   # a crash or Ctrl-C: say what was kept, then let the error through
            if not isinstance(e, SystemExit) and checkpoints:
                run_id, n_done, path = checkpoints[-1]
                console.print(f"[bold red]RUN INTERRUPTED ({type(e).__name__}).[/bold red] The {n_done} benchmark(s) finished before it are saved as an "
                              f"[bold]INCOMPLETE[/bold] record ({run_id}) at {path}. It cannot be compared or gated; rerun to get a complete run.")
            raise
        saved_path = store.save_run(artifact)
        if artifact.is_failed:
            console.print(f"[bold red]RUN FAILED[/bold red] ({artifact.run_id}, job {artifact.job_name}): {artifact.failure_reason}")
            console.print(f"[dim]A partial record of {len(artifact.item_results)} item(s) was saved to {saved_path}. It is not added "
                          f"to the run list or the leaderboard, and `compare` / `gate` will not accept it.[/dim]")
            failed_models.append(m_id)
            continue
        console.print(f"[bold green]? Completed run {artifact.run_id}[/bold green] ({artifact.run_mode.upper()}, job {artifact.job_name}, "
                      f"scorer v{artifact.scorer_version}) -> saved to [dim]{saved_path}[/dim]")
        if artifact.served_model_id and artifact.served_model_id != artifact.model_id:
            console.print(f"[dim]Served as {artifact.served_model_id} (requested {artifact.model_id})[/dim]")
        if artifact.model_substituted:
            console.print(f"[bold yellow]SUBSTITUTE ROUTE:[/bold yellow] {artifact.route_note or 'the served model is not the exact requested variant'} "
                          f"Results are for {artifact.served_model_id}, not the exact requested model.")
        if artifact.is_standin:
            console.print("[bold yellow]STAND-IN RUN:[/bold yellow] pipeline validation only, not a result for the real models.")

        # Display Summary Table
        t = Table(title=f"Results for {artifact.model_name} - {artifact.run_mode.upper()} (Overall: {artifact.overall_accuracy*100:.1f}%)")
        t.add_column("Benchmark", style="bold")
        t.add_column("Domain")
        t.add_column("Score / Total")
        t.add_column("Accuracy (95% CI)")
        t.add_column("Truncated / No answer / API error")
        t.add_column("Reliability")
        t.add_column("Mean Latency")

        for b_id, s in artifact.benchmark_summaries.items():
            ci_str = f"[{s.ci_lower*100:.1f}%, {s.ci_upper*100:.1f}%]"
            reliability = f"[bold red]UNRELIABLE ({s.unreliable_fraction:.0%})[/bold red]" if s.unreliable else "ok"
            t.add_row(s.benchmark_name, s.domain, f"{s.correct_samples}/{s.total_samples}", f"{s.accuracy*100:.1f}% {ci_str}", f"{s.truncated_count} / {s.no_answer_count} / {s.api_error_count}", reliability, f"{s.mean_latency:.2f}s")
        console.print(t)
        if artifact.unreliable_benchmarks:
            console.print(f"[bold red]UNRELIABLE BENCHMARK(S):[/bold red] {', '.join(artifact.unreliable_benchmarks)}: more than "
                          f"{UNRELIABLE_BENCHMARK_THRESHOLD:.0%} of their items were api_error, truncated or no_answer_found, "
                          f"so those rows do not measure the model fairly, even if the run as a whole is under {UNRELIABLE_THRESHOLD:.0%}.")
        if artifact.truncated_count or artifact.no_answer_count or artifact.api_error_count:
            console.print(f"[bold yellow]Note:[/bold yellow] {artifact.truncated_count} response(s) were cut off by max_tokens, "
                          f"{artifact.no_answer_count} had no extractable answer and {artifact.api_error_count} calls failed "
                          f"(api_error, after any retries). All are counted as incorrect.")
        if artifact.unreliable:
            flagged = artifact.api_error_count + artifact.truncated_count + artifact.no_answer_count
            console.print(f"[bold red]UNRELIABLE RUN:[/bold red] {flagged} of {len(artifact.item_results)} items "
                          f"({artifact.unreliable_fraction:.0%}) were api_error, truncated or no_answer_found (limit {UNRELIABLE_THRESHOLD:.0%}). "
                          f"This run's accuracy is not a fair measure of the model; fix the cause and rerun.")

    if failed_models:
        console.print(f"[bold red]{len(failed_models)} model(s) FAILED and were stopped early:[/bold red] {', '.join(failed_models)}")
        sys.exit(1)

def _refuse_unusable_runs(console_, *runs):
    """compare and gate only accept COMPLETED runs: a FAILED (stopped early) or INCOMPLETE (never finished) run is refused."""
    unusable = [r for r in runs if not r.is_usable]
    if unusable:
        for r in unusable:
            console_.print(f"[bold red]Error: run {r.run_id} is {r.status.upper()} and cannot be used here:[/bold red] {r.failure_reason}")
        sys.exit(1)

@cli.command("compare")
@click.option("--run-a", required=True, help="First run ID")
@click.option("--run-b", required=True, help="Second run ID")
@click.option("--output-md", default=None, help="Save report to markdown file")
@click.option("--output-html", default=None, help="Save report to HTML file")
def compare_cmd(run_a, run_b, output_md, output_html):
    """Compare two evaluation runs (score deltas, flips, McNemar test)."""
    store = RunStore("runs")
    ra = store.get_run(run_a)
    rb = store.get_run(run_b)

    if not ra or not rb:
        console.print("[bold red]Error: One or both run IDs could not be found.[/bold red]")
        sys.exit(1)
    _refuse_unusable_runs(console, ra, rb)

    comp = DiffEngine.compare_runs(ra, rb)

    console.print(Panel(f"[bold cyan]Comparing Run A:[/bold cyan] {ra.model_name} vs [bold cyan]Run B:[/bold cyan] {rb.model_name}\n"
                        f"Run mode: A={ra.run_mode.upper()}, B={rb.run_mode.upper()}\n"
                        f"Job: A={ra.job_name}, B={rb.job_name}" + ("  [bold yellow]STAND-IN: pipeline validation only[/bold yellow]" if ra.is_standin or rb.is_standin else "") + "\n"
                        f"Served model: A={comp['served_model_id_a']} ({'SUBSTITUTE' if ra.model_substituted else 'exact'}), "
                        f"B={comp['served_model_id_b']} ({'SUBSTITUTE' if rb.model_substituted else 'exact'})\n"
                        + (f"[bold yellow]SUBSTITUTE ROUTE (A):[/bold yellow] {ra.route_note}\n" if ra.model_substituted and ra.route_note else "")
                        + (f"[bold yellow]SUBSTITUTE ROUTE (B):[/bold yellow] {rb.route_note}\n" if rb.model_substituted and rb.route_note else "")
                        + f"Truncated / no-answer / api_error (counted incorrect): A={ra.truncated_count} / {ra.no_answer_count} / {ra.api_error_count}, "
                        f"B={rb.truncated_count} / {rb.no_answer_count} / {rb.api_error_count}\n"
                        f"Reliability: A={'UNRELIABLE' if ra.unreliable else 'ok'} ({ra.unreliable_fraction:.0%}), "
                        f"B={'UNRELIABLE' if rb.unreliable else 'ok'} ({rb.unreliable_fraction:.0%})\n"
                        f"Unreliable benchmarks (over {UNRELIABLE_BENCHMARK_THRESHOLD:.0%} of their items flagged): "
                        f"A={', '.join(ra.unreliable_benchmarks) or 'none'}; B={', '.join(rb.unreliable_benchmarks) or 'none'}\n"
                        f"Scorer version: A={ra.scorer_version}{' (RESCORED)' if ra.is_rescored else ''}, "
                        f"B={rb.scorer_version}{' (RESCORED)' if rb.is_rescored else ''}"
                        + ("  [bold yellow]different scorer versions: scores are not directly comparable[/bold yellow]" if ra.scorer_version != rb.scorer_version else "") + "\n"
                        f"Overall Accuracy: {ra.overall_accuracy*100:.1f}% -> {rb.overall_accuracy*100:.1f}% "
                        f"([bold]{comp['delta_accuracy']*100:+.1f}%[/bold])\n"
                        f"Latency Delta: {comp['delta_latency']:+.2f}s | Flips (B wins: {comp['b_only_correct_count']}, A wins: {comp['a_only_correct_count']})\n"
                        f"McNemar p-value: {comp['mcnemar']['p_value']} (Significant: {comp['mcnemar']['significant']})",
                        title="nel compare Results", border_style="green"))

    # Benchmarks delta table
    t = Table(title="Benchmark Score Deltas")
    t.add_column("Benchmark", style="bold")
    t.add_column(ra.model_name)
    t.add_column(rb.model_name)
    t.add_column("Delta", style="bold")
    t.add_column("Reliability")

    for b_id, d in comp["benchmarks"].items():
        color = "green" if d["delta"] > 0 else ("red" if d["delta"] < 0 else "white")
        flags = [f"{side} UNRELIABLE ({d['unreliable_fraction_' + side.lower()]:.0%})" for side in ("A", "B") if d["unreliable_" + side.lower()]]
        t.add_row(d["benchmark_name"], f"{d['accuracy_a']*100:.1f}%", f"{d['accuracy_b']*100:.1f}%", f"[{color}]{d['delta']*100:+.1f}%[/{color}]",
                  f"[bold red]{'; '.join(flags)}[/bold red]" if flags else "ok")
    console.print(t)

    if output_md:
        md = ReportGenerator.generate_markdown(comp)
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(md)
        console.print(f"[green]Markdown report exported to {output_md}[/green]")

    if output_html:
        html = ReportGenerator.generate_html(comp)
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html)
        console.print(f"[green]HTML report exported to {output_html}[/green]")

@cli.command("gate")
@click.option("--baseline", required=True, help="Baseline run ID")
@click.option("--candidate", required=True, help="Candidate run ID")
@click.option("--min-acc", default=0.65, type=float, help="Minimum overall accuracy threshold")
@click.option("--max-reg", default=0.03, type=float, help="Maximum allowed regression delta")
def gate_cmd(baseline, candidate, min_acc, max_reg):
    """Quality Gate check (GO / NO-GO evaluation for release readiness)."""
    store = RunStore("runs")
    ra = store.get_run(baseline)
    rb = store.get_run(candidate)

    if not ra or not rb:
        console.print("[bold red]Error: Baseline or candidate run not found.[/bold red]")
        sys.exit(1)
    _refuse_unusable_runs(console, ra, rb)

    gate = StatisticsEngine.evaluate_quality_gate(
        baseline_summary={"overall_accuracy": ra.overall_accuracy},
        candidate_summary={"overall_accuracy": rb.overall_accuracy},
        min_accuracy=min_acc,
        max_regression=max_reg
    )

    color = "bold green" if gate["status"] == "PASS" else "bold red"
    console.print(Panel(f"Decision: [{color}]{gate['status']}[/{color}]\n"
                        f"Run mode: baseline={ra.run_mode.upper()}, candidate={rb.run_mode.upper()}\n"
                        f"Served model: baseline={ra.served_model_id or ra.model_id} ({'SUBSTITUTE' if ra.model_substituted else 'exact'}), "
                        f"candidate={rb.served_model_id or rb.model_id} ({'SUBSTITUTE' if rb.model_substituted else 'exact'})\n"
                        f"Reliability: baseline={'UNRELIABLE' if ra.unreliable else 'ok'}, candidate={'UNRELIABLE' if rb.unreliable else 'ok'}"
                        + ("  [bold red]a decision on an UNRELIABLE run is not safe[/bold red]" if ra.unreliable or rb.unreliable else "") + "\n"
                        f"Unreliable benchmarks: baseline={', '.join(ra.unreliable_benchmarks) or 'none'}; candidate={', '.join(rb.unreliable_benchmarks) or 'none'}\n"
                        f"Scorer version: baseline={ra.scorer_version}, candidate={rb.scorer_version}"
                        + ("  [bold yellow]different scorer versions: not directly comparable[/bold yellow]" if ra.scorer_version != rb.scorer_version else "") + "\n"
                        f"Baseline Accuracy: {gate['baseline_accuracy']*100:.1f}%\n"
                        f"Candidate Accuracy: {gate['candidate_accuracy']*100:.1f}% (Delta: {gate['accuracy_delta']*100:+.1f}%)\n"
                        f"Rationale: {gate['rationale']}",
                        title="NeMo Evaluator Quality Gate (nel gate)", border_style="cyan"))

if __name__ == "__main__":
    cli()
