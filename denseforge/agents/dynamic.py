"""Dynamic Agent Spawning — on-demand agent creation."""
from loguru import logger


class DynamicAgentManager:
    """Create specialized agents on demand."""

    AGENT_TYPES = {
        "researcher": "You are a research specialist. Analyze and find information.",
        "coder": "You are a coding specialist. Write and review code.",
        "critic": "You are a critic. Evaluate quality and find issues.",
        "planner": "You are a planner. Break tasks into steps.",
        "writer": "You are a writer. Create clear, structured content.",
    }

    def __init__(self, llm_fn=None):
        self.llm_fn = llm_fn
        self._agents: dict[str, dict] = {}
        self._spawn_count = 0

    def spawn(self, agent_type: str, task: str) -> dict:
        if agent_type not in self.AGENT_TYPES:
            agent_type = "researcher"
        agent_id = f"agent-{self._spawn_count}"
        self._spawn_count += 1
        self._agents[agent_id] = {"type": agent_type, "task": task, "status": "active"}
        logger.info(f"Spawned {agent_type} agent: {agent_id}")
        return {"agent_id": agent_id, "type": agent_type, "system_prompt": self.AGENT_TYPES[agent_type]}

    async def execute(self, agent_id: str, prompt: str) -> str:
        agent = self._agents.get(agent_id)
        if not agent:
            return "Agent not found"
        if self.llm_fn:
            system = self.AGENT_TYPES.get(agent["type"], "")
            return self.llm_fn(f"{system}\n\nTask: {prompt}")
        return f"[STUB] {agent['type']} response to: {prompt[:100]}"

    def efficiency_report(self) -> dict:
        return {"total_spawned": self._spawn_count, "active": len(self._agents)}
