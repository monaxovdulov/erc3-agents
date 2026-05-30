from __future__ import annotations

from openai import OpenAI
from erc3 import ERC3, TaskInfo

from .execution import ExecutionEngine
from .profiling import build_task_profile
from .tools import bootstrap_state
from .models import TaskIntent, TaskProfile


class AgentCore:
    def __init__(self, model: str, client: OpenAI, core: ERC3):
        self.model = model
        self.client = client
        self.core = core
        self.engine = ExecutionEngine(client, model, core)

    def run_task(self, task: TaskInfo) -> None:
        try:
            profile, profile_stats = build_task_profile(self.client, self.model, task, self.core)
        except Exception as e:  # pragma: no cover - defensive fallback
            print("  Task profiling failed, using conservative defaults:", e)
            profile = TaskProfile(
                intent=TaskIntent.OTHER,
                domain="unknown",
                rbac_sensitivity="high",
                needs_whoami=True,
                needs_rulebook=False,
                max_steps=4,
            )
            profile_stats = (0, 0, 0.0)
        state, store_api = bootstrap_state(self.core, task, profile)

        print(
            f"  TaskProfile: intent={profile.intent}, domain={profile.domain}, "
            f"rbac={profile.rbac_sensitivity}, max_steps={profile.max_steps}"
        )
        print(
            f"  Profiling tokens: prompt={profile_stats[0]}, "
            f"completion={profile_stats[1]}, time={profile_stats[2]:.3f}s"
        )

        self.engine.run(task, profile, state, store_api)
