"""
Test agent for Airline RM environment.

Runs a single task using GPT, logs all interactions to a JSONL trajectory file
for inspection and debugging.
"""

import json
import asyncio
import os
import time
from datetime import datetime

from openai import AsyncOpenAI
from openreward import OpenReward


async def main():
    or_client = OpenReward()
    oai_client = AsyncOpenAI()

    MODEL_NAME = "gpt-5.2"
    ENV_NAME = "GeneralReasoning/airlinerm"
    SPLIT = "train"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OR_API_KEY = os.getenv("OPENREWARD_API_KEY")

    # Trajectory log file
    trajectory_file = f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"Logging trajectory to: {trajectory_file}")

    def log_event(event_type: str, data: dict):
        """Append an event to the JSONL trajectory file."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **data,
        }
        with open(trajectory_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    environment = or_client.environments.get(name=ENV_NAME)
    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks in '{SPLIT}' split")
    print(f"Available tools: {[t['function']['name'] for t in tools]}")

    # Run only the first task for testing
    task = tasks[0]
    print(f"\nStarting task: {task.task_spec}")

    log_event("task_start", {"task_spec": task.task_spec, "model": MODEL_NAME})

    rollout = or_client.rollout.create(
        run_name=ENV_NAME.split("/")[-1] + "_test",
        rollout_name="test_run",
        environment=ENV_NAME,
        split=SPLIT,
        task_spec=task.task_spec,
    )

    async with environment.session(task=task, secrets={"api_key": OR_API_KEY}) as session:
        prompt = await session.get_prompt()
        input_list = [{"role": "user", "content": prompt[0].text}]
        finished = False
        turn = 0

        log_event("prompt", {"content": prompt[0].text[:500] + "..."})
        rollout.log_openai_response(message=input_list[0], is_finished=finished)

        while not finished:
            turn += 1
            t0 = time.time()

            response = await oai_client.responses.create(
                model=MODEL_NAME,
                tools=tools,
                input=input_list,
            )

            elapsed = time.time() - t0
            rollout.log_openai_response(response.output[-1])
            input_list += response.output

            # Log model response
            for item in response.output:
                if hasattr(item, "type"):
                    if item.type == "message":
                        content = getattr(item, "content", "")
                        if isinstance(content, list) and content:
                            text = content[0].text if hasattr(content[0], "text") else str(content[0])
                        else:
                            text = str(content)
                        log_event("model_message", {
                            "turn": turn,
                            "text": text[:1000],
                            "elapsed_s": round(elapsed, 2),
                        })
                    elif item.type == "function_call":
                        log_event("function_call", {
                            "turn": turn,
                            "name": item.name,
                            "arguments": str(item.arguments)[:1000],
                            "elapsed_s": round(elapsed, 2),
                        })

            # Process tool calls
            for item in response.output:
                if hasattr(item, "type") and item.type == "function_call":
                    tool_result = await session.call_tool(
                        item.name, json.loads(str(item.arguments))
                    )

                    reward = tool_result.reward
                    finished = tool_result.finished

                    output_text = tool_result.blocks[0].text if tool_result.blocks else ""

                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": output_text,
                    })
                    rollout.log_openai_response(
                        input_list[-1], reward=reward, is_finished=finished
                    )

                    # Log tool result
                    log_event("tool_result", {
                        "turn": turn,
                        "tool": item.name,
                        "reward": reward,
                        "finished": finished,
                        "output_preview": output_text[:500],
                    })

                    print(f"Turn {turn}: {item.name} -> reward={reward:.4f}, finished={finished}")

                    if finished:
                        print(f"\nFINISHED! Final reward: {reward:.4f}")
                        log_event("task_finished", {
                            "turn": turn,
                            "final_reward": reward,
                            "total_turns": turn,
                        })
                        break

        print(f"\nTotal turns: {turn}")
        print(f"Trajectory saved to: {trajectory_file}")


if __name__ == "__main__":
    asyncio.run(main())
