"""
cli.py
------
Concept: CLI AS THE CI ENTRY POINT

Everything above is a library. This file is the thin wrapper that makes it
runnable from a terminal or a GitHub Actions step. The most important detail
for CI is the EXIT CODE:
  - sys.exit(0) -> CI step passes, pipeline continues
  - sys.exit(1) -> CI step fails, pipeline stops / PR gets blocked

That's the entire mechanism by which "regression testing for prompts"
plugs into a real CI/CD system — no special GitHub integration needed,
just a shell command that can fail.
"""
from __future__ import annotations

import sys
import click
from dotenv import load_dotenv

from .dataset import load_dataset
from .runner import run_dataset
from .pipeline import score_case, build_run_summary
from .baseline import load_baseline, save_baseline, save_results
from .report import diff_against_baseline, format_report, build_slack_payload, post_to_slack

load_dotenv()


@click.group()
def cli():
    """LLM Eval / Regression Testing Harness — CI/CD for prompts."""
    pass


@cli.command()
@click.option("--dataset", required=True, help="Path to golden dataset YAML.")
@click.option("--target", required=True, help="'module:function' or an HTTP URL for the app under test.")
@click.option("--baseline-path", default="baseline.json", help="Path to baseline results JSON.")
@click.option("--results-path", default="results/latest.json", help="Where to save this run's results.")
@click.option("--no-judge", is_flag=True, help="Skip the LLM-as-judge scorer (faster, no API cost).")
@click.option("--no-hallucination", is_flag=True, help="Skip hallucination checking.")
@click.option("--slack-webhook", default=None, help="Optional Slack incoming webhook URL to post the report to.")
@click.option("--fail-on-regression/--no-fail-on-regression", default=True,
              help="Exit non-zero if any case regressed vs baseline (default: on).")
def run(dataset, target, baseline_path, results_path, no_judge, no_hallucination, slack_webhook, fail_on_regression):
    """Run the golden dataset against TARGET, score it, and diff vs baseline."""
    cases = load_dataset(dataset)
    click.echo(f"Loaded {len(cases)} cases from {dataset}")

    click.echo(f"Running against target: {target} ...")
    run_outputs = run_dataset(cases, target)
    outputs_by_id = {o.case_id: o for o in run_outputs}

    click.echo("Scoring ...")
    records = [
        score_case(case, outputs_by_id[case.id], use_judge=not no_judge, use_hallucination=not no_hallucination)
        for case in cases
    ]

    run_summary = build_run_summary(records, target=target, dataset_path=dataset)
    save_results(run_summary, results_path)

    baseline = load_baseline(baseline_path)
    diff = diff_against_baseline(records, baseline["results"] if baseline else None)
    if baseline is None:
        click.echo(f"(No baseline found at {baseline_path} — run 'baseline update' after reviewing this run.)")

    report = format_report(run_summary, diff)
    click.echo("\n" + report)

    if slack_webhook:
        try:
            post_to_slack(slack_webhook, build_slack_payload(run_summary, diff))
            click.echo("\nPosted report to Slack.")
        except Exception as e:  # noqa: BLE001
            click.echo(f"\n[warning] failed to post to Slack: {e}")

    any_failed = any(not r["passed"] for r in records)
    should_fail = any_failed or (fail_on_regression and diff["has_regression"])
    sys.exit(1 if should_fail else 0)


@cli.group()
def baseline():
    """Manage the saved baseline used for regression comparisons."""
    pass


@baseline.command("update")
@click.option("--results-path", default="results/latest.json", help="Results file to promote to baseline.")
@click.option("--baseline-path", default="baseline.json", help="Where to write the new baseline.")
def baseline_update(results_path, baseline_path):
    """Promote a previous run's results to be the new baseline. Do this deliberately after reviewing scores."""
    data = load_baseline(results_path)
    if data is None:
        click.echo(f"No results file found at {results_path}. Run 'eval-harness run' first.")
        sys.exit(1)
    save_baseline(data, baseline_path)
    click.echo(f"Baseline updated: {baseline_path} (from {results_path}, {len(data['results'])} cases)")


@cli.command()
@click.argument("run_a")
@click.argument("run_b")
def diff(run_a, run_b):
    """Diff two arbitrary results JSON files against each other (not just current-vs-baseline)."""
    a = load_baseline(run_a)
    b = load_baseline(run_b)
    if a is None or b is None:
        click.echo("One or both files not found.")
        sys.exit(1)
    d = diff_against_baseline(b["results"], a["results"])
    click.echo(format_report(b, d))
    sys.exit(1 if d["has_regression"] else 0)


if __name__ == "__main__":
    cli()
