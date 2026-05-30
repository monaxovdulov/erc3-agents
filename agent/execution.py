from __future__ import annotations

import time
from typing import Type

from openai import OpenAI
from erc3 import ERC3, ApiException, TaskInfo
from erc3 import erc3 as dev

from .context import build_messages_for_step
from .models import (
    AgentState,
    TaskProfile,
    TaskIntent,
    ExecutionStep,
    ProjectStep,
    EmployeeStep,
    WikiStep,
    TimeStep,
    CustomerStep,
    DomainStep,
)
from .security import SecurityGuard
from .tools import summarise_tool_result


class ExecutionEngine:
    def __init__(self, client: OpenAI, model: str, core: ERC3):
        self.client = client
        self.model = model
        self.core = core

    def _response_schema(self, profile: TaskProfile) -> Type[DomainStep]:
        match profile.domain:
            case "projects":
                return ProjectStep
            case "employees":
                return EmployeeStep
            case "wiki":
                return WikiStep
            case "time":
                return TimeStep
            case "customers":
                return CustomerStep
            case _:
                return ExecutionStep

    def run(self, task: TaskInfo, profile: TaskProfile, state: AgentState, store_api) -> None:
        guard = SecurityGuard()
        max_steps = profile.max_steps
        schema = self._response_schema(profile)
        responded = False
        logged_time = False

        for _ in range(max_steps):
            state.steps_done += 1
            messages = build_messages_for_step(state)

            started = time.time()
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                response_format=schema,
                messages=messages,
                max_completion_tokens=2048,
            )
            duration = time.time() - started

            usage = completion.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            if usage:
                try:
                    self.core.log_llm(
                        task_id=task.task_id,
                        completion=(
                            f"ExecutionStep: prompt={prompt_tokens}, "
                            f"completion={completion_tokens}, duration={duration:.3f}s"
                        ),
                        model=self.model,
                        duration_sec=duration,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cached_prompt_tokens=getattr(
                            getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0
                        ),
                    )
                except ApiException as e:
                    print("  [telemetry] log_llm failed for ExecutionStep:", e.detail)

            step = completion.choices[0].message.parsed

            if profile.domain == "time" and getattr(step, "task_completed", False) and not logged_time:
                if profile.intent != TaskIntent.DATE_QUESTION:
                    state.facts.append("Cannot finalize yet: no time entry logged. Use Req_LogTimeEntry first.")
                    state.last_tool_summary = state.facts[-1]
                    continue

            if getattr(step, "task_completed", False):
                resp = step.respond or dev.Req_ProvideAgentResponse(
                    outcome="error_internal",
                    message="Model declared completion but no structured response was provided.",
                    links=[],
                )
                responded = True
                self._dispatch_final(store_api, resp, state)
                break

            tool = getattr(step, "tool", None)
            if tool is None:
                deny_resp = dev.Req_ProvideAgentResponse(
                    outcome="none_unsupported",
                    message="Requested operation is not supported in this system (no matching tool for the request).",
                    links=[],
                )
                responded = True
                self._dispatch_final(store_api, deny_resp, state)
                break

            # Auto-fill actors for write operations when possible before permission checks.
            self._fill_actor_fields(tool, state)

            if not guard.can_execute(tool, state):
                deny_resp = guard.build_denial_response(tool, state)
                responded = True
                self._dispatch_final(store_api, deny_resp, state)
                break

            try:
                result = store_api.dispatch(tool)
                summary = summarise_tool_result(tool, result)
                state.last_tool_summary = summary
                state.facts.append(summary)
                if isinstance(tool, dev.Req_LogTimeEntry):
                    logged_time = True
            except ApiException as e:
                err = getattr(e, "api_error", None)
                detail = getattr(err, "error", None) or e.detail
                err_summary = f"{tool.__class__.__name__} failed: {detail}"
                failure_resp = dev.Req_ProvideAgentResponse(
                    outcome="error_internal",
                    message=err_summary,
                    links=[],
                )
                state.last_tool_summary = err_summary
                state.facts.append(err_summary)
                responded = True
                self._dispatch_final(store_api, failure_resp, state)
                break
        else:
            # Step budget exhausted without completion
            if not responded:
                responded = True
                deny_resp = guard.build_generic_failure(state)
                self._dispatch_final(store_api, deny_resp, state)

        # Safety net: ensure a final response exists.
        if not responded:
            deny_resp = guard.build_generic_failure(state)
            self._dispatch_final(store_api, deny_resp, state)
            responded = True

    def _dispatch_final(self, store_api, resp: dev.Req_ProvideAgentResponse, state: AgentState) -> None:
        try:
            store_api.dispatch(resp)
            state.last_tool_summary = resp.message
            state.facts.append(resp.message)
        except ApiException as e:
            err = getattr(e, "api_error", None)
            detail = getattr(err, "error", None) or e.detail
            state.last_tool_summary = f"Dispatch failed: {detail}"
            state.facts.append(state.last_tool_summary)

    def _fill_actor_fields(self, tool: object, state: AgentState) -> None:
        current_user = (state.whoami_raw or {}).get("current_user")
        if not current_user:
            return

        if isinstance(tool, dev.Req_LogTimeEntry):
            if getattr(tool, "employee", None) is None:
                tool.employee = current_user
            if getattr(tool, "logged_by", None) is None:
                tool.logged_by = current_user
            if getattr(tool, "billable", None) is None:
                tool.billable = True
            if getattr(tool, "status", None) in (None, ""):
                tool.status = "submitted"
            if getattr(tool, "work_category", None) in (None, ""):
                tool.work_category = "general"
            if getattr(tool, "notes", None) in (None, ""):
                tool.notes = "Logged via agent"

        if isinstance(tool, dev.Req_UpdateTimeEntry):
            if getattr(tool, "changed_by", None) is None:
                tool.changed_by = current_user

        if isinstance(tool, (dev.Req_UpdateEmployeeInfo, dev.Req_UpdateProjectStatus, dev.Req_UpdateProjectTeam, dev.Req_UpdateWiki)):
            if getattr(tool, "changed_by", None) is None:
                tool.changed_by = current_user
