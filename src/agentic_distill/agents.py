"""Teacher agent variants (registered with tau2's registry on import)."""

from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT, LLMAgent, create_llm_agent
from tau2.registry import registry

RATIONALE_INSTRUCTION = (
    "Before every tool call, write one short sentence (at most 25 words) in the same message "
    "saying what you are about to do and why. This overrides the rule above that you cannot "
    "send a message and make a tool call at the same time."
)


class RationaleLLMAgent(LLMAgent):
    """LLMAgent whose system prompt asks for a one-sentence rationale before each tool call."""

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=f"{AGENT_INSTRUCTION}\n{RATIONALE_INSTRUCTION}",
        )


def create_rationale_llm_agent(tools, domain_policy, **kwargs):
    return RationaleLLMAgent(
        tools=tools, domain_policy=domain_policy,
        llm=kwargs.get("llm"), llm_args=kwargs.get("llm_args"),
        additional_stop_tool_names=kwargs.get("additional_stop_tool_names", frozenset()),
    )


if "rationale_llm_agent" not in registry.get_agents():
    registry.register_agent_factory(create_rationale_llm_agent, "rationale_llm_agent")
