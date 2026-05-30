from __future__ import annotations

import time
from typing import Tuple

from openai import OpenAI
from erc3 import ERC3, ApiException, TaskInfo

from .models import TaskProfile


def build_task_profile(
    client: OpenAI,
    model: str,
    task: TaskInfo,
    core: ERC3,
) -> Tuple[TaskProfile, Tuple[int, int, float]]:
    """
    Single-shot call to classify the task and propose execution settings.
    Returns (profile, (prompt_tokens, completion_tokens, duration_sec)).
    """
    system = (
        "You are an intent classifier for an internal enterprise agent. "
        "Classify the user task, determine the business domain, and propose "
        "a concise execution budget. Keep max_steps small but sufficient."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task.task_text},
    ]

    started = time.time()
    completion = client.beta.chat.completions.parse(
        model=model,
        response_format=TaskProfile,
        messages=messages,
        max_completion_tokens=512,
    )
    duration = time.time() - started

    usage = completion.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0

    if usage:
        completion_summary = (
            f"TaskProfile: prompt={prompt_tokens}, completion={completion_tokens}, "
            f"duration={duration:.3f}s"
        )
        try:
            core.log_llm(
                task_id=task.task_id,
                completion=completion_summary,
                model=model,
                duration_sec=duration,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_prompt_tokens=getattr(
                    getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0
                ),
            )
        except ApiException as e:
            print("  [telemetry] log_llm failed for TaskProfile:", e.detail)

    return completion.choices[0].message.parsed, (prompt_tokens or 0, completion_tokens or 0, duration)
