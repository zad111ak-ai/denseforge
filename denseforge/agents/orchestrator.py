"""MultiAgentOrchestrator — orchestrate multiple agents for complex task solving."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Callable

from loguru import logger


# ---------------------------------------------------------------------------
# Agent roles
# ---------------------------------------------------------------------------

class _AgentRole:
    """Lightweight agent that performs a specific sub-task."""

    def __init__(self, name: str, llm_fn: Callable[[str], str]):
        self.name = name
        self.llm_fn = llm_fn

    async def act(self, task: str, context: str = "") -> dict:
        """Execute the agent's portion of the work asynchronously."""
        loop = asyncio.get_event_loop()
        prompt = (
            f"You are the **{self.name}** agent.\n\n"
            f"TASK: {task}\n\n"
            f"CONTEXT:\n{context}\n\n"
            "Provide your response as a JSON object with a 'result' key "
            "and any supporting 'reasoning' or 'evidence' keys."
        )
        result = await loop.run_in_executor(None, self.llm_fn, prompt)
        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            parsed = {"result": result}
        return {
            "agent": self.name,
            "response": parsed,
            "timestamp": time.time(),
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class MultiAgentOrchestrator:
    """Coordinate multiple specialised agents to solve complex queries.

    Parameters
    ----------
    llm_fn:
        A callable ``(prompt: str) -> str`` for the underlying LLM.
    denseforge:
        A reference to the main DenseForge instance (used for retrieval and
        memory access).  May be ``None`` for standalone usage.
    """

    def __init__(self, llm_fn: Callable[[str], str], denseforge=None):
        self.llm_fn = llm_fn
        self.denseforge = denseforge
        self._agents: dict[str, _AgentRole] = {}
        self._history: list[dict] = []
        logger.info("MultiAgentOrchestrator initialised")

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------
    def register_agent(self, name: str) -> _AgentRole:
        """Register a named agent with the orchestrator."""
        agent = _AgentRole(name, self.llm_fn)
        self._agents[name] = agent
        logger.debug("Registered agent: {name}", name=name)
        return agent

    def _get_default_agents(self) -> list[str]:
        """Return a sensible set of default agent names."""
        defaults = [
            "retrieval_specialist",
            "reasoning_agent",
            "synthesis_agent",
        ]
        for name in defaults:
            if name not in self._agents:
                self.register_agent(name)
        return defaults

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    async def _plan(self, task: str, user_id: str | None = None) -> dict:
        """Use the LLM to decompose *task* into sub-tasks for each agent."""
        agent_names = list(self._agents.keys()) or self._get_default_agents()

        planning_prompt = (
            "You are a task planner.  Break the following task into sub-tasks "
            "that can be assigned to specialised agents.\n\n"
            f"AVAILABLE AGENTS: {', '.join(agent_names)}\n\n"
            f"TASK: {task}\n\n"
            "Respond with a JSON list of objects, each with 'agent' and "
            "'sub_task' keys."
        )
        loop = asyncio.get_event_loop()
        plan_raw = await loop.run_in_executor(None, self.llm_fn, planning_prompt)

        try:
            plan = json.loads(plan_raw)
            if not isinstance(plan, list):
                plan = [{"agent": agent_names[0], "sub_task": task}]
        except (json.JSONDecodeError, TypeError):
            # Fallback: assign the entire task to the first agent
            plan = [{"agent": agent_names[0], "sub_task": task}]

        logger.info(
            "Task decomposed into {n} sub-tasks",
            n=len(plan),
        )
        return {
            "plan": plan,
            "task_id": str(uuid.uuid4()),
            "user_id": user_id,
        }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    async def _execute_plan(self, plan: dict) -> list[dict]:
        """Run all sub-tasks concurrently and collect results."""
        sub_tasks = plan.get("plan", [])
        if not sub_tasks:
            return []

        tasks = []
        for item in sub_tasks:
            agent_name = item.get("agent", "")
            sub_task_text = item.get("sub_task", "")
            agent = self._agents.get(agent_name)
            if agent is None:
                logger.warning("Agent '{name}' not found; using fallback", name=agent_name)
                agent = self._agents[next(iter(self._agents))]  # first available
            tasks.append(agent.act(sub_task_text))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        outcomes: list[dict] = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger.error("Agent {idx} failed: {err}", idx=i, err=res)
                outcomes.append({
                    "agent": sub_tasks[i].get("agent", "unknown"),
                    "response": {"result": f"Agent failed: {res}"},
                    "timestamp": time.time(),
                    "error": str(res),
                })
            else:
                outcomes.append(res)

        return outcomes

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------
    async def _synthesize(self, task: str, results: list[dict]) -> dict:
        """Combine sub-agent results into a final answer."""
        synthesis_parts = []
        for r in results:
            agent_name = r.get("agent", "unknown")
            resp = r.get("response", {})
            synthesis_parts.append(
                f"[{agent_name}]: {json.dumps(resp, default=str)}"
            )

        synthesis_prompt = (
            "You are the synthesis agent.  Combine the following sub-agent "
            f"results into a coherent answer to: \"{task}\"\n\n"
            "SUB-AGENT RESULTS:\n"
            + "\n".join(synthesis_parts)
            + "\n\nProvide a unified JSON response with 'answer', "
            "'confidence' (0-1), and 'supporting_agents' keys."
        )
        loop = asyncio.get_event_loop()
        synth_raw = await loop.run_in_executor(None, self.llm_fn, synthesis_prompt)

        try:
            synthesis = json.loads(synth_raw)
        except (json.JSONDecodeError, TypeError):
            synthesis = {"answer": synth_raw, "confidence": 0.5, "supporting_agents": []}

        return synthesis

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def solve_complex_task(self, query: str, user_id: str | None = None) -> dict:
        """End-to-end complex task solving: plan → execute → synthesise.

        Parameters
        ----------
        query:
            The complex user query.
        user_id:
            Optional user identifier for memory / history tracking.

        Returns
        -------
        dict
            ``task_id``, ``answer``, ``confidence``, ``sub_results``, ``plan``.
        """
        logger.info("Solving complex task: {q}", q=query[:120])

        plan = await self._plan(query, user_id)
        results = await self._execute_plan(plan)
        synthesis = await self._synthesize(query, results)

        outcome = {
            "task_id": plan["task_id"],
            "answer": synthesis.get("answer", ""),
            "confidence": synthesis.get("confidence", 0.0),
            "supporting_agents": synthesis.get("supporting_agents", []),
            "sub_results": results,
            "plan": plan["plan"],
        }

        self._history.append(outcome)
        logger.info(
            "Task {tid} solved ({n} agents, conf={c:.2f})",
            tid=plan["task_id"][:8],
            n=len(results),
            c=outcome["confidence"],
        )
        return outcome

    def plan_and_act(self, task: str, user_id: str | None = None) -> dict:
        """Synchronous wrapper around :meth:`solve_complex_task`.

        Runs the async pipeline in a new event loop if called from a
        non-async context.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an async loop — cannot use asyncio.run
            # Fall back to a dedicated thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.solve_complex_task(task, user_id))
                return future.result(timeout=120)
        else:
            return asyncio.run(self.solve_complex_task(task, user_id))

    def history(self) -> list[dict]:
        """Return a copy of the orchestrator's task history."""
        return list(self._history)
