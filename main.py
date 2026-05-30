#!/usr/bin/env python
from __future__ import annotations

import textwrap
from typing import Optional

from openai import OpenAI
from erc3 import ERC3

from agent import AgentCore

MODEL_ID = "gpt-4o"


def main() -> None:
    client = OpenAI()
    core = ERC3()  # ERC3_API_KEY and ERC3_BASE_URL are read from the environment

    session = core.start_session(
        benchmark="erc3-dev",
        workspace="my",
        name=f"Hybrid Agent ({MODEL_ID})",
        architecture="Hybrid Two-Phase Agent",
        flags=["compete_accuracy"],
    )
    print(f"Started session: {session.session_id}")

    status = core.session_status(session.session_id)
    print(f"Tasks in session: {len(status.tasks)}")

    agent = AgentCore(model=MODEL_ID, client=client, core=core)

    for task in status.tasks:
        print("=" * 60)
        print(f"Task {task.task_id} ({task.spec_id}):")
        print(textwrap.indent(task.task_text, "  "))

        core.start_task(task)
        try:
            agent.run_task(task)
        except Exception as e:  # pragma: no cover - defensive catch
            print("  Agent error:", e)

        result = core.complete_task(task)
        _print_eval(result.eval)

    submit = core.submit_session(session.session_id)
    print("Session submitted, status:", submit.status)


def _print_eval(eval_result: Optional[object]) -> None:
    if not eval_result:
        print("  SCORE: n/a (no eval)")
        return
    print(f"  SCORE: {eval_result.score}")
    print(textwrap.indent(eval_result.logs, "    "))


if __name__ == "__main__":
    main()
