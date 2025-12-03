import json
import sys
import time
from pathlib import Path
from typing import Annotated, List, Union, Literal

import erc3
from annotated_types import MaxLen, MinLen
from pydantic import BaseModel, Field
from erc3 import erc3 as dev, ApiException, TaskInfo, ERC3, Erc3Client
from openai import OpenAI

client = OpenAI()

class NextStep(BaseModel):
    current_state: str
    # we'll use only the first step, discarding all the rest.
    plan_remaining_steps_brief: Annotated[List[str], MinLen(1), MaxLen(5)] =  Field(..., description="explain your thoughts on how to accomplish - what steps to execute")
    # now let's continue the cascade and check with LLM if the task is done
    task_completed: bool
    # Routing to one of the tools to execute the first remaining step
    # if task is completed, model will pick ReportTaskCompletion
    function: Union[
        dev.Req_ProvideAgentResponse,
        dev.Req_ListProjects,
        dev.Req_ListEmployees,
        dev.Req_ListCustomers,
        dev.Req_GetCustomer,
        dev.Req_GetEmployee,
        dev.Req_GetProject,
        dev.Req_GetTimeEntry,
        dev.Req_SearchProjects,
        dev.Req_SearchEmployees,
        dev.Req_LogTimeEntry,
        dev.Req_SearchTimeEntries,
        dev.Req_SearchCustomers,
        dev.Req_UpdateTimeEntry,
        dev.Req_UpdateProjectTeam,
        dev.Req_UpdateProjectStatus,
        dev.Req_UpdateEmployeeInfo,
        dev.Req_TimeSummaryByProject,
        dev.Req_TimeSummaryByEmployee,
    ] = Field(..., description="execute first remaining step")



CLI_RED = "\x1B[31m"
CLI_GREEN = "\x1B[32m"
CLI_BLUE = "\x1B[34m"
CLI_CLR = "\x1B[0m"



def distill_rules(model: str, api: Erc3Client) -> str:

    who = api.who_am_i()
    context_id = who.wiki_sha1

    loc = Path(f"context_{context_id}.json")

    Category = Literal["applies_to_guests", "applies_to_users", "other"]

    class Rule(BaseModel):
        why_relevant_summary: str = Field(...)
        category: Category = Field(...)
        compact_rule: str

    class DistillWikiRules(BaseModel):
        company_name: str
        rules: List[Rule]

    if  not loc.exists():




        schema = json.dumps(NextStep.model_json_schema())
        prompt = f"""
    Carefully review the wiki below and identify most important security/scoping/data rules that will be highly relevant for the agent or user that are automating APIs of this company.

    Rules must be compact RFC-style, ok to use pseudo code for compactness. They will be used by an agent that operates following APIs: {schema}
    """.strip()


        # pull wiki

        for path in api.list_wiki().paths:
            content = api.load_wiki(path)

            prompt += f"\n---- start of {path} ----\n\n{content}\n\n ---- end of {path} ----\n"


        messages = [{ "role": "system", "content": prompt}]

        response = client.beta.chat.completions.parse(model=model, response_format=DistillWikiRules, messages=messages)
        distilled = response.choices[0].message.parsed
        loc.write_text(distilled.model_dump_json(indent=2))

    else:
        distilled = DistillWikiRules.model_validate_json(loc.read_text())

    about = api.who_am_i()
    prompt = f"""You are AI Chatbot automating {distilled.company_name}

Use available tools to execute task from the current user.

To confirm project access - get or find project (and get after finding)
When updating entry - fill all fields to keep with old values from being erased
When task is done or can't be done - Req_ProvideAgentResponse.

# Rules
"""

    relevant_categories: List[Category] = ["other"]
    if about.is_public:
        relevant_categories.append("applies_to_guests")
    else:
        relevant_categories.append("applies_to_users")

    for r in distilled.rules:
        if r.category in relevant_categories:
            prompt += f"\n- {r.compact_rule}"

    # append at the end to keep rules in context cache

    if about.is_public:
        prompt += "\n# /whoami Current user is GUEST (/whoami)"
    else:
        employee = api.get_employee(about.current_user).employee
        dump = employee.model_dump_json()
        prompt += f"\n# /whoami Current user is {employee.name}\n{dump}"

        # dump project brief
    return prompt



def run_agent(model: str, api: ERC3, task: TaskInfo):

    erc_client = api.get_erc_client(task)


    system_prompt = distill_rules(model, erc_client)


    # log will contain conversation context for the agent within task
    log = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task.task_text},
    ]

    # let's limit number of reasoning steps by 20, just to be safe
    for i in range(20):
        step = f"step_{i + 1}"
        print(f"Next {step}... ", end="")

        started = time.time()

        completion = client.beta.chat.completions.parse(
            model=model,
            response_format=NextStep,
            messages=log,
            max_completion_tokens=20384,
        )

        api.log_llm(
            task_id=task.task_id,
            model=model, # must match slug from OpenRouter
            duration_sec=time.time() - started,
            usage=completion.usage,
        )

        job = completion.choices[0].message.parsed

          # print next sep for debugging
        print(job.plan_remaining_steps_brief[0], f"\n  {job.function}")

        # Let's add tool request to conversation history as if OpenAI asked for it.
        # a shorter way would be to just append `job.model_dump_json()` entirely
        log.append({
            "role": "assistant",
            "content": job.plan_remaining_steps_brief[0],
            "tool_calls": [{
                "type": "function",
                "id": step,
                "function": {
                    "name": job.function.__class__.__name__,
                    "arguments": job.function.model_dump_json(),
                }}]
        })

        # now execute the tool by dispatching command to our handler
        try:
            result = erc_client.dispatch(job.function)
            txt = result.model_dump_json(exclude_none=True, exclude_unset=True)
            print(f"{CLI_GREEN}OUT{CLI_CLR}: {txt}")
        except ApiException as e:
            txt = e.detail
            # print to console as ascii red
            print(f"{CLI_RED}ERR: {e.api_error.error}{CLI_CLR}")

            # if SGR wants to finish, then quit loop
        if isinstance(job.function, dev.Req_ProvideAgentResponse):
            print(f"{CLI_BLUE}agent {job.function.outcome}{CLI_CLR}. Summary:\n{job.function.message}")

            for link in job.function.links:
                print(f"  - link {link.kind}: {link.id}")

            break

        # and now we add results back to the convesation history, so that agent
        # we'll be able to act on the results in the next reasoning step.
        log.append({"role": "tool", "content": txt, "tool_call_id": step})
