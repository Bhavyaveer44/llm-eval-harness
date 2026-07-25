# LLM Eval / Regression Testing Harness

CI/CD for prompts. Run a golden dataset against your AI app on every change,
score the outputs, and fail the build if something regressed.

## Why this exists

Traditional software has deterministic tests: same input, same output,`assert`. 
LLM apps don't work that way, the same prompt can return different (but equally valid) wording every time. 
Most teams shipping LLM features either have no automated testing at all, 
or eyeball a few examples before deploying. That's the gap this fills.


## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY

# Run without judge/hallucination scoring first (free, fast, tests plumbing)
python -m eval_harness.cli run \
  --dataset golden_dataset.example.yaml \
  --target target_app_example:answer \
  --no-judge --no-hallucination

# Once you're happy with a run, promote it to the baseline
python -m eval_harness.cli baseline update

# Now run the full pipeline (uses your Anthropic API key for judge + hallucination)
python -m eval_harness.cli run \
  --dataset golden_dataset.example.yaml \
  --target target_app_example:answer

# Optional: post the report to Slack
python -m eval_harness.cli run --dataset ... --target ... \
  --slack-webhook https://hooks.slack.com/services/...
```

## Break it yourself (the fun part)

1. **Force a regression.** Edit `target_app_example.py`'s system prompt (see
   the comment at the bottom of the file) to encourage confident, made-up
   answers. Rerun — watch `hallucinated` flip to `true` and judge scores
   drop below threshold on the policy cases, and the diff column show
   `REGRESSED`.

2. **Break the similarity scorer.** Change a `reference` in the YAML to
   something in a totally different topic. Watch the score crash toward 0
   even though the app's real answer might be fine — this demonstrates why
   similarity alone isn't enough, and why the judge score exists as a second
   signal.

3. **Break the judge.** Write a deliberately vague `judge_rubric` (e.g. "is
   this a good response?") vs a specific one, and compare how consistent
   the scores are across repeated runs. This is the core tension in
   LLM-as-judge design: rubric specificity vs a written-a-rubric-for-every-
   possible-case blowup.

4. **Simulate a flaky app.** Make `target_app_example.answer` randomly
   `raise Exception("timeout")` some % of the time. Watch `pipeline.py`'s
   error handling mark those cases as automatic fails without crashing the
   whole run — this is what makes the harness safe to run unattended in CI.

5. **Add an HTTP target.** Stand up a tiny Flask/FastAPI server wrapping
   your real app, then run with `--target http://localhost:8000/generate`
   instead of a Python path — no code import needed, closer to how you'd
   eval a real staging deployment.

## Wiring into CI (GitHub Actions example)

```yaml
# .github/workflows/eval.yml
name: LLM Eval
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt
      - run: |
          python -m eval_harness.cli run \
            --dataset golden_dataset.example.yaml \
            --target target_app_example:answer \
            --slack-webhook ${{ secrets.SLACK_WEBHOOK_URL }}
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

A regression or failing case makes this step fail red on the PR — that's
the whole "CI/CD for prompts" pitch, working end to end.

## Known limitations (worth knowing, worth mentioning in an interview)

- TF-IDF similarity is a stand-in for real embeddings — swap it for
  production use (see comment in `similarity.py`).
- LLM-as-judge isn't perfectly deterministic even at temperature 0; that's
  why `REGRESSION_TOLERANCE` exists rather than treating any score change
  as significant.
- The hallucination scorer fails "open" (reports no hallucination) if the
  judge API call itself errors — deliberate, so an infra hiccup doesn't
  show up as a false content-quality regression. Consider whether you'd
  want fail-closed behavior for a stricter safety-critical use case.
- Matching cases between runs is by `id` string — renaming a case ID makes
  the diff treat it as REMOVED + NEW instead of tracking its history.
