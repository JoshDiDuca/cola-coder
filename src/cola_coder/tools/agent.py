"""Agent loop for iterative tool-augmented generation.

The agent generates, detects tool calls, executes tools,
feeds results back, and continues generating. This enables
multi-step problem solving.

Max iterations prevents infinite loops. Each iteration:
1. Generate response
2. Check for tool calls
3. If found: execute, feed result, go to 1
4. If not: return final response
"""

from dataclasses import dataclass, field
from typing import Any

from cola_coder.tools.executor import ToolExecutor, ToolResult
from cola_coder.tools.formatter import format_tool_result, parse_tool_call
from cola_coder.tools.registry import ToolRegistry


@dataclass
class AgentStep:
    """One step in the agent loop."""

    iteration: int
    generation: str
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class AgentResult:
    """Result of an agent run."""

    final_response: str
    steps: list[AgentStep]
    total_iterations: int
    total_tool_calls: int

    @property
    def used_tools(self) -> bool:
        return self.total_tool_calls > 0


class AgentLoop:
    """Iterative agent that generates code with tool access.

    The agent can call tools during generation to verify its work,
    search for relevant code, check types, and run tests.

    Flow:
    1. Generate with tool definitions in system prompt
    2. Parse any tool calls from output
    3. Execute tools safely
    4. Feed results back as tool messages
    5. Continue generating
    6. Stop when no more tool calls or max iterations reached
    """

    def __init__(
        self,
        generator: Any,
        tokenizer: Any,
        registry: ToolRegistry | None = None,
        executor: ToolExecutor | None = None,
        max_iterations: int = 5,
    ):
        """
        Args:
            generator: CodeGenerator instance
            tokenizer: CodeTokenizer instance
            registry: Tool registry (uses defaults if None)
            executor: Tool executor (uses defaults if None)
            max_iterations: Max generate-tool-generate cycles
        """
        self.generator = generator
        self.tokenizer = tokenizer
        self.registry = registry or ToolRegistry()
        self.executor = executor or ToolExecutor()
        self.max_iterations = max_iterations

    def run(
        self,
        prompt: str,
        system_prompt: str = "",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> AgentResult:
        """Run the agent loop.

        Args:
            prompt: User's prompt
            system_prompt: Additional system prompt
            max_new_tokens: Max tokens per generation
            temperature: Sampling temperature

        Returns:
            AgentResult with final response and step history
        """
        # Build system prompt with tool definitions
        tool_prompt = self.registry.get_system_prompt()
        if system_prompt and tool_prompt:
            full_system = f"{system_prompt}\n\n{tool_prompt}"
        elif tool_prompt:
            full_system = tool_prompt
        else:
            full_system = system_prompt

        # Build initial prompt
        current_prompt = f"{full_system}\n\n{prompt}" if full_system else prompt

        steps: list[AgentStep] = []
        total_tool_calls = 0

        for iteration in range(self.max_iterations):
            # Generate
            response = self.generator.generate(
                current_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )

            # Parse tool calls
            calls = parse_tool_call(response)

            step = AgentStep(
                iteration=iteration,
                generation=response,
                tool_calls=calls,
            )

            if not calls:
                # No tool calls — we're done
                steps.append(step)
                break

            # Execute tools
            for call in calls:
                tool_name = call.get("name", "")
                arguments = call.get("arguments", {})

                result = self.executor.execute(tool_name, arguments)
                step.tool_results.append(result)
                total_tool_calls += 1

                # Format result for next iteration
                result_text = format_tool_result(tool_name, result.output, result.success)
                current_prompt += f"\n\n{response}\n\n{result_text}\n\nContinue:"

            steps.append(step)

        # Get final response (last generation)
        final = steps[-1].generation if steps else ""

        return AgentResult(
            final_response=final,
            steps=steps,
            total_iterations=len(steps),
            total_tool_calls=total_tool_calls,
        )
