from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field

from erc3 import TaskInfo
from erc3 import erc3 as dev


class TaskIntent(str, Enum):
    SEARCH = "search"
    UPDATE = "update"
    AUTH_CHECK = "auth_check"
    DATE_QUESTION = "date_question"
    IMPOSSIBLE = "impossible"
    AMBIGUOUS = "ambiguous"
    OTHER = "other"


class TaskProfile(BaseModel):
    intent: TaskIntent
    domain: Literal["employees", "projects", "wiki", "customers", "time", "system", "unknown"]
    rbac_sensitivity: Literal["low", "medium", "high"]
    needs_whoami: bool = True
    needs_rulebook: bool = False
    max_steps: int = Field(ge=1, le=10, default=6)
    required_tools: List[str] = Field(default_factory=list)
    answer_format_hint: str = ""
    notes: str = ""


ProjectTools = Union[
    dev.Req_ListProjects,
    dev.Req_SearchProjects,
    dev.Req_GetProject,
    dev.Req_UpdateProjectStatus,
    dev.Req_UpdateProjectTeam,
    dev.Req_TimeSummaryByProject,
]

EmployeeTools = Union[
    dev.Req_ListEmployees,
    dev.Req_SearchEmployees,
    dev.Req_GetEmployee,
    dev.Req_UpdateEmployeeInfo,
    dev.Req_TimeSummaryByEmployee,
]

CustomerTools = Union[
    dev.Req_ListCustomers,
    dev.Req_SearchCustomers,
    dev.Req_GetCustomer,
]

WikiTools = Union[
    dev.Req_ListWiki,
    dev.Req_LoadWiki,
    dev.Req_SearchWiki,
    dev.Req_UpdateWiki,
]

TimeTools = Union[
    dev.Req_LogTimeEntry,
    dev.Req_UpdateTimeEntry,
    dev.Req_GetTimeEntry,
    dev.Req_SearchTimeEntries,
]

AnyTool = Union[ProjectTools, EmployeeTools, CustomerTools, WikiTools, TimeTools]


class ExecutionStep(BaseModel):
    thought: str = Field(..., description="Short reasoning", max_length=300)
    task_completed: bool = Field(..., description="True if agent can respond now")
    respond: Optional[dev.Req_ProvideAgentResponse] = Field(
        None,
        description="Final response payload. Include links to referenced entities (projects/employees/customers) in links[].",
    )
    tool: Optional[AnyTool] = Field(
        None, description="API call to execute when task_completed is false"
    )


class ProjectStep(BaseModel):
    thought: str = Field(..., description="Short reasoning", max_length=300)
    task_completed: bool = Field(..., description="True if agent can respond now")
    respond: Optional[dev.Req_ProvideAgentResponse] = Field(
        None,
        description="Final response payload. Include links to referenced entities (projects/employees/customers) in links[].",
    )
    tool: Optional[Union[ProjectTools, CustomerTools, TimeTools]] = Field(
        None, description="Project-focused tool call"
    )


class EmployeeStep(BaseModel):
    thought: str = Field(..., description="Short reasoning", max_length=300)
    task_completed: bool = Field(..., description="True if agent can respond now")
    respond: Optional[dev.Req_ProvideAgentResponse] = Field(
        None,
        description="Final response payload. Include links to referenced entities (projects/employees/customers) in links[].",
    )
    tool: Optional[Union[EmployeeTools, ProjectTools, TimeTools]] = Field(
        None, description="Employee-focused tool call"
    )


class WikiStep(BaseModel):
    thought: str = Field(..., description="Short reasoning", max_length=300)
    task_completed: bool = Field(..., description="True if agent can respond now")
    respond: Optional[dev.Req_ProvideAgentResponse] = Field(
        None,
        description="Final response payload. Include links to referenced entities (projects/employees/customers) in links[].",
    )
    tool: Optional[WikiTools] = Field(None, description="Wiki tool call")


class TimeStep(BaseModel):
    thought: str = Field(..., description="Short reasoning", max_length=300)
    task_completed: bool = Field(..., description="True if agent can respond now")
    respond: Optional[dev.Req_ProvideAgentResponse] = Field(
        None,
        description="Final response payload. Include links to referenced entities (projects/employees/customers) in links[].",
    )
    tool: Optional[
        Union[
            dev.Req_LogTimeEntry,
            dev.Req_SearchTimeEntries,
            dev.Req_GetProject,
            dev.Req_SearchProjects,
            dev.Req_GetEmployee,
            dev.Req_SearchEmployees,
        ]
    ] = Field(
        None,
        description="Time tracking flow: primarily use Req_LogTimeEntry. Use search/get only to resolve IDs.",
    )


class CustomerStep(BaseModel):
    thought: str = Field(..., description="Short reasoning", max_length=300)
    task_completed: bool = Field(..., description="True if agent can respond now")
    respond: Optional[dev.Req_ProvideAgentResponse] = Field(
        None,
        description="Final response payload. Include links to referenced entities (projects/employees/customers) in links[].",
    )
    tool: Optional[Union[CustomerTools, ProjectTools]] = Field(
        None, description="Customer tool call"
    )


DomainStep = Union[ProjectStep, EmployeeStep, WikiStep, TimeStep, CustomerStep, ExecutionStep]


@dataclass
class AgentState:
    task: TaskInfo
    profile: TaskProfile
    whoami_raw: Optional[dict] = None
    current_user_summary: Optional[str] = None
    rulebook_summary: Optional[str] = None
    facts: List[str] = field(default_factory=list)
    last_tool_summary: Optional[str] = None
    steps_done: int = 0
