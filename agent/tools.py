from __future__ import annotations

from typing import Optional, Tuple

from erc3 import ERC3, ApiException, TaskInfo
from erc3 import erc3 as dev

from .models import AgentState, TaskProfile


def bootstrap_state(core: ERC3, task: TaskInfo, profile: TaskProfile) -> Tuple[AgentState, dev.Erc3Client]:
    store_api = core.get_erc_client(task)
    state = AgentState(task=task, profile=profile)

    try:
        whoami = store_api.who_am_i()
        state.whoami_raw = whoami.model_dump()
        if whoami.is_public:
            summary = "public/guest access"
        else:
            parts = [whoami.current_user or "unknown user"]
            if whoami.department:
                parts.append(f"dept={whoami.department}")
            if whoami.location:
                parts.append(f"loc={whoami.location}")
            summary = ", ".join(parts)
        state.current_user_summary = summary
        state.facts.append(f"Auth: {summary}")
        if whoami.today:
            state.facts.append(f"Today: {whoami.today}")
    except ApiException as e:
        state.facts.append(f"whoami failed: {e.detail}")

    if profile.needs_rulebook or profile.rbac_sensitivity == "high":
        rulebook = _load_rulebook_summary(store_api)
        if rulebook:
            state.rulebook_summary = rulebook
            state.facts.append(f"Rulebook: {rulebook}")

    return state, store_api


def _load_rulebook_summary(store_api) -> Optional[str]:
    try:
        search = store_api.search_wiki(query_regex="rulebook|rbac|policy")
        candidates = [r.path for r in (search.results or [])]
        if not candidates:
            listing = store_api.list_wiki()
            candidates = [
                p for p in listing.paths if "rule" in p.lower() or "policy" in p.lower()
            ]
        if not candidates:
            return None
        target = candidates[0]
        content = store_api.load_wiki(file=target).content
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        snippet = "; ".join(lines[:5])
        return f"{target}: {snippet[:300]}"
    except ApiException:
        return None


def summarise_tool_result(tool, result) -> str:
    cls = tool.__class__.__name__
    try:
        if isinstance(tool, dev.Req_ListProjects):
            projects = result.projects or []
            head = ", ".join(f"{p.id}:{p.name}({p.status})" for p in projects[:3])
            return f"ListProjects -> {len(projects)} projects; {head}"

        if isinstance(tool, dev.Req_SearchProjects):
            projects = result.projects or []
            head = ", ".join(f"{p.id}:{p.name}({p.status})" for p in projects[:3])
            return f"SearchProjects -> {len(projects)} results; {head}"

        if isinstance(tool, dev.Req_GetProject):
            proj = getattr(result, "project", None)
            if proj:
                team_size = len(proj.team) if proj.team else 0
                team_desc = ", ".join(f"{m.employee}:{m.role}({m.time_slice:.0f}%)" for m in (proj.team or [])[:5])
                return (
                    f"Project {proj.id}: {proj.name}, status={proj.status}, "
                    f"customer={proj.customer}, team={team_size} [{team_desc}]. "
                    f"Use project link id={proj.id}."
                )
            return "Project not found"

        if isinstance(tool, dev.Req_UpdateProjectStatus):
            return f"Updated project {tool.id} status -> {tool.status}"

        if isinstance(tool, dev.Req_UpdateProjectTeam):
            return f"Updated project {tool.id} team size -> {len(tool.team)}"

        if isinstance(tool, dev.Req_ListEmployees):
            employees = result.employees or []
            head = ", ".join(f"{e.id}:{e.name}({e.department})" for e in employees[:3])
            return f"ListEmployees -> {len(employees)} employees; {head}"

        if isinstance(tool, dev.Req_SearchEmployees):
            employees = result.employees or []
            head = ", ".join(f"{e.id}:{e.name}({e.department})" for e in employees[:3])
            return f"SearchEmployees -> {len(employees)} results; {head}"

        if isinstance(tool, dev.Req_GetEmployee):
            emp = getattr(result, "employee", None)
            if emp:
                return (
                    f"Employee {emp.id}: {emp.name}, dept={emp.department}, "
                    f"loc={emp.location}, salary={emp.salary}, email={emp.email}. Use employee link id={emp.id}."
                )
            return "Employee not found"

        if isinstance(tool, dev.Req_UpdateEmployeeInfo):
            return f"Updated employee {tool.employee}"

        if isinstance(tool, dev.Req_ListCustomers):
            companies = result.companies or []
            head = ", ".join(f"{c.id}:{c.name}({c.deal_phase})" for c in companies[:3])
            return f"ListCustomers -> {len(companies)} companies; {head}"

        if isinstance(tool, dev.Req_SearchCustomers):
            companies = result.companies or []
            head = ", ".join(f"{c.id}:{c.name}({c.deal_phase})" for c in companies[:3])
            return f"SearchCustomers -> {len(companies)} results; {head}"

        if isinstance(tool, dev.Req_GetCustomer):
            company = getattr(result, "company", None)
            if company:
                return (
                    f"Customer {company.id}: {company.name}, "
                    f"status={company.high_level_status}, phase={company.deal_phase}, "
                    f"primary_contact={company.primary_contact_name}<{company.primary_contact_email}>"
                )
            return "Customer not found"

        if isinstance(tool, dev.Req_LogTimeEntry):
            return f"Logged time for employee {tool.employee} on {tool.date} ({tool.hours}h)"

        if isinstance(tool, dev.Req_UpdateTimeEntry):
            return f"Updated time entry {tool.id} ({tool.hours}h on {tool.date})"

        if isinstance(tool, dev.Req_GetTimeEntry):
            entry = getattr(result, "entry", None)
            if entry:
                return (
                    f"Time entry {tool.id}: {entry.hours}h on {entry.date}, "
                    f"project={entry.project}, status={entry.status}"
                )
            return "Time entry not found"

        if isinstance(tool, dev.Req_SearchTimeEntries):
            entries = result.entries or []
            return (
                f"SearchTimeEntries -> {len(entries)} entries; "
                f"total_hours={result.total_hours:.2f}"
            )

        if isinstance(tool, dev.Req_TimeSummaryByProject):
            summaries = result.summaries or []
            head = ", ".join(
                f"{s.project}:{s.total_hours:.1f}h(billable {s.billable_hours:.1f})"
                for s in summaries[:3]
            )
            return f"TimeSummaryByProject -> {len(summaries)} rows; {head}"

        if isinstance(tool, dev.Req_TimeSummaryByEmployee):
            summaries = result.summaries or []
            head = ", ".join(
                f"{s.employee}:{s.total_hours:.1f}h(billable {s.billable_hours:.1f})"
                for s in summaries[:3]
            )
            return f"TimeSummaryByEmployee -> {len(summaries)} rows; {head}"

        if isinstance(tool, (dev.Req_ListWiki, dev.Req_SearchWiki, dev.Req_LoadWiki, dev.Req_UpdateWiki)):
            return f"Wiki op {cls} succeeded."

    except Exception:
        return f"{cls} executed (summary failed)."

    return f"{cls} executed successfully."
