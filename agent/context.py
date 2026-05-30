from __future__ import annotations

from typing import Any, Dict, List

from .models import AgentState, TaskIntent


def build_messages_for_step(state: AgentState) -> List[Dict[str, Any]]:
    remaining = max(state.profile.max_steps - state.steps_done, 0)
    system_parts = [
        "You are a fast, security-aware enterprise assistant.",
        "Respect RBAC and company policies. If an action is not permitted, reply with Req_ProvideAgentResponse and outcome=denied_security.",
        "At each turn either propose a single tool call or finalize with Req_ProvideAgentResponse, never both.",
        "Exactly one Req_ProvideAgentResponse must be sent per task.",
        "When referencing specific entities, include AgentLink entries in links (e.g., project link uses kind=project and the project id).",
        "If the request is ambiguous or missing identifiers, respond with outcome=none_clarification_needed asking for the needed detail.",
        "CEO helene_stutz is allowed to approve salary updates; authenticated users may read project/customer contact info (including emails). Guests must not access internal data.",
        "Project leads can update their project's status; non-leads must refuse.",
        "Before changing project status, fetch project details to confirm the lead role.",
        f"Step budget remaining: {remaining} (stop early if ready).",
    ]
    if state.current_user_summary:
        system_parts.append(f"Current user: {state.current_user_summary}")
    if state.rulebook_summary:
        system_parts.append(f"RBAC summary: {state.rulebook_summary}")
    if state.profile.answer_format_hint:
        system_parts.append(f"Preferred answer format: {state.profile.answer_format_hint}")
    if state.profile.domain == "time" and state.profile.intent != TaskIntent.DATE_QUESTION:
        system_parts.append(
            "For time tracking tasks, you must log time via Req_LogTimeEntry before finalizing. Use current user for 'me' as employee/logged_by, set billable=True, include project id, date (today/yesterday), hours, and fill work_category/status/notes with sensible defaults (e.g., work_category='general', status='submitted'). Do not send final response until logging succeeds."
        )

    system = "\n".join(system_parts)

    facts_block = "\n".join(f"- {f}" for f in state.facts[-5:]) if state.facts else "none yet"
    last_tool = state.last_tool_summary or "no tool calls yet"

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": state.task.task_text},
        {"role": "assistant", "content": f"TaskProfile: {state.profile.model_dump_json()}"},
        {"role": "assistant", "content": f"Known facts:\n{facts_block}\nLast tool: {last_tool}"},
    ]
    return messages
