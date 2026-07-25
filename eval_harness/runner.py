"""
Concept: THE SYSTEM UNDER TEST (SUT)

The harness needs to be app-agnostic, it shouldn't care if the "app" is a LangChain pipeline,
a raw API call, or a FastAPI endpoint. So we support two ways of plugging the app:

1. Python target: "mymodule.mysubmodule:my_function"
   We dynamically import the module and grab the function by name.
   The function's contract must be: fn(prompt: str, context: str | None) -> str

2. HTTP target: a URL, "http://localhost:8000/generate"
   We POST {"prompt": ..., "context": ...} and expect back {"output": "..."}.
   This is how you'd hook up a real deployed service without importing its code at all,
   useful for CI running against a staging deployment.

This separation is the same idea as a test runner (pytest) not caring what
the code does internally, it just needs a callable interface.
"""
from __future__ import annotations

import importlib
import time
import requests
from dataclasses import dataclass
from .dataset import EvalCase


@dataclass
class RunOutput:
    case_id: str
    output: str
    latency_ms: float
    error: str | None = None


def _load_python_target(target: str):
    """target looks like 'module.path:function_name'"""
    if ":" not in target:
        raise ValueError(
            "Python target must look like 'module.path:function_name', "
            f"got '{target}'"
        )
    module_path, func_name = target.split(":", 1)
    module = importlib.import_module(module_path)
    fn = getattr(module, func_name, None)
    if fn is None:
        raise AttributeError(f"'{func_name}' not found in module '{module_path}'")
    return fn


def _call_http_target(url: str, prompt: str, context: str | None, timeout: int = 60) -> str:
    resp = requests.post(url, json={"prompt": prompt, "context": context}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if "output" not in data:
        raise ValueError(f"HTTP target must return JSON with an 'output' key. Got: {data}")
    return data["output"]


def run_dataset(cases: list[EvalCase], target: str) -> list[RunOutput]:
    """
    Execute every case against the target app and collect raw outputs.
    This step is deliberately separate from scoring — you might want to run
    once and score multiple ways, or re-score cached outputs without paying
    for the app calls again.
    """
    is_http = target.startswith("http://") or target.startswith("https://")
    fn = None if is_http else _load_python_target(target)

    results: list[RunOutput] = []
    for case in cases:
        start = time.time()
        try:
            if is_http:
                output = _call_http_target(target, case.prompt, case.context)
            else:
                output = fn(case.prompt, case.context)
            error = None
        except Exception as e:  # noqa: BLE001 - we want to record ANY failure, not crash the run
            output = ""
            error = f"{type(e).__name__}: {e}"
        latency_ms = (time.time() - start) * 1000
        results.append(RunOutput(case_id=case.id, output=output, latency_ms=latency_ms, error=error))
    return results
