"""Stop conditions for the hand-built ReAct loop (Week 7 · Module 4).

The one thing this loop must never do is run forever. Each test forces a
different way a model can fail to converge, and checks the loop stops safely
rather than hanging or burning an unbounded number of calls.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from app.agents import claim_handover_agent as agent_mod
from app.claims.tools import ToolResult
from app.core.enums import UserRole
from app.llm.base import Completion
from app.security.authz import AuthSubject

SUBJECT = AuthSubject(
    user_id=uuid.uuid4(), tenant_id="default", role=UserRole.AGENT, policy_holder_id=None
)


class QueueLLM:
    """Returns queued completions in order; repeats the last once exhausted."""

    def __init__(self, texts: list[str], *, input_tokens: int = 10) -> None:
        self._texts = texts
        self._input_tokens = input_tokens
        self.calls = 0

    async def complete(self, messages, **kwargs) -> Completion:  # noqa: ARG002
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        return Completion(text=text, model="fake", input_tokens=self._input_tokens, output_tokens=5)


@pytest.fixture
def patch_llm(monkeypatch):
    def apply(fake: QueueLLM):
        monkeypatch.setattr(agent_mod, "get_llm", lambda: fake)

    return apply


@pytest.fixture
def patch_tools(monkeypatch):
    """Stub run_tool so the loop never touches a real database."""

    async def fake_run_tool(name, args, *, session, subject):  # noqa: ARG001
        return ToolResult(ok=True, observation=f"observed {name}({args})")

    monkeypatch.setattr(agent_mod, "run_tool", fake_run_tool)


async def test_repeated_action_stops_the_loop(patch_llm, patch_tools):
    same_step = (
        '{"thought": "check", "action": "get_claim_status", '
        '"action_input": {"claim_number": "CLM-1"}}'
    )
    fake = QueueLLM([same_step])
    patch_llm(fake)

    result = await agent_mod.run(session=None, subject=SUBJECT, claim_number="CLM-1")

    assert result.stop_reason == "repeated_action"
    assert fake.calls == 2  # proposes the same action twice, then stops
    assert result.tool_calls == 1  # only ran it once


async def test_never_finishing_stops_at_max_steps(patch_llm, patch_tools):
    # A distinct claim number each turn so the repeated-action guard never fires
    # first - this is testing the *step* budget specifically.
    steps = [
        (
            f'{{"thought": "t{i}", "action": "get_claim_status", '
            f'"action_input": {{"claim_number": "CLM-{i}"}}}}'
        )
        for i in range(agent_mod.MAX_STEPS + 3)
    ]
    fake = QueueLLM(steps)
    patch_llm(fake)

    result = await agent_mod.run(session=None, subject=SUBJECT, claim_number="CLM-0")

    assert result.stop_reason == "max_steps_exceeded"
    assert result.tool_calls == agent_mod.MAX_STEPS
    assert fake.calls == agent_mod.MAX_STEPS
    assert "max_steps_exceeded" in result.answer


async def test_malformed_output_stops_without_crashing(patch_llm, patch_tools):
    fake = QueueLLM(["not json at all"])
    patch_llm(fake)

    result = await agent_mod.run(session=None, subject=SUBJECT, claim_number="CLM-1")

    assert result.stop_reason == "malformed_output"
    assert result.tool_calls == 0
    assert result.answer  # still produces something, not an exception


async def test_token_budget_stops_before_a_second_call(patch_llm, patch_tools):
    step = (
        '{"thought": "t", "action": "get_claim_status", '
        '"action_input": {"claim_number": "CLM-1"}}'
    )
    fake = QueueLLM([step], input_tokens=agent_mod.MAX_TOTAL_TOKENS + 1)
    patch_llm(fake)

    result = await agent_mod.run(session=None, subject=SUBJECT, claim_number="CLM-1")

    assert result.stop_reason == "token_budget_exceeded"
    assert fake.calls == 1  # the second decide call never happens
    assert result.tool_calls == 1  # but the first tool call it already asked for did


async def test_time_budget_stops_before_a_second_call(patch_llm, patch_tools, monkeypatch):
    """Mirrors the token-budget test: the decide call itself takes real wall time,
    so the *next* iteration's pre-check is what has to catch it."""

    class SlowLLM(QueueLLM):
        async def complete(self, messages, **kwargs):  # noqa: ARG002
            await asyncio.sleep(0.05)
            return await super().complete(messages)

    monkeypatch.setattr(agent_mod, "MAX_SECONDS", 0.01)
    step = (
        '{"thought": "t", "action": "get_claim_status", '
        '"action_input": {"claim_number": "CLM-1"}}'
    )
    fake = SlowLLM([step])
    patch_llm(fake)

    result = await agent_mod.run(session=None, subject=SUBJECT, claim_number="CLM-1")

    assert result.stop_reason == "time_budget_exceeded"
    assert fake.calls == 1  # the second decide call never happens
    assert result.tool_calls == 1  # but the first tool call it already asked for did


async def test_clause_search_refuses_without_a_prior_claim_lookup(patch_llm, patch_tools):
    """date_of_loss picks which wording version is in force - the tool must
    never run unfiltered just because the model skipped get_claim_status."""
    steps = [
        '{"thought": "t", "action": "search_policy_clause", '
        '"action_input": {"clause_ref": "4.11"}}',
        '{"thought": "ok", "action": "finish", "final_answer": "done"}',
    ]
    fake = QueueLLM(steps)
    patch_llm(fake)

    result = await agent_mod.run(session=None, subject=SUBJECT, claim_number="CLM-1")

    assert result.stop_reason == "finished"
    assert result.tool_calls == 1
    assert result.steps[0].ok is False
    assert "get_claim_status first" in result.steps[0].observation


async def test_finish_stops_immediately(patch_llm, patch_tools):
    fake = QueueLLM(['{"thought": "done", "action": "finish", "final_answer": "all good"}'])
    patch_llm(fake)

    result = await agent_mod.run(session=None, subject=SUBJECT, claim_number="CLM-1")

    assert result.stop_reason == "finished"
    assert result.answer == "all good"
    assert result.tool_calls == 0
    assert fake.calls == 1  # never asks a second time once it can finish


async def test_unknown_tool_name_is_an_observation_not_a_crash():
    from app.claims.tools import run_tool

    result = await run_tool("delete_everything", {}, session=None, subject=SUBJECT)

    assert result.ok is False
    assert "Unknown tool" in result.observation
