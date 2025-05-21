import json
from aiocache import cached
from litellm import acompletion, completion_cost
from tenacity import retry, stop_after_attempt, stop_before_delay

from askametric.utils import setup_logger

llm_call_logger = setup_logger("LLM_call")


def enforce_json_format(system_message: str) -> str:
    """
    Modify the system message to JSON format in the LLM response by modifying
    the input system message.
    """

    force_to_json_prompt = """
    Respond ONLY in a python parsable JSON format.
    Do not include any extra text, explanations, or markdown formatting.

    This is wrong -
    ```json
    {
        "key_1": "value_1",
        "key_2": "value_2"
    }
    ```

    This is correct -
    {
        "key_1": "value_1",
        "key_2": "value_2"
    }
    """
    # Append the JSON enforcement prompt only if it's not already present
    if force_to_json_prompt.strip() not in system_message:
        modified_sys_message = f"{system_message}\n{force_to_json_prompt}"

    return modified_sys_message


@cached(ttl=60 * 60 * 24)
@retry(
    stop=(stop_after_attempt(3) | stop_before_delay(10)),
)
async def ask_llm_json(
    prompt: str,
    system_message: str,
    llm: str = "gpt-4o",
    temperature: float = 0.1,
    api_key: str | None = None,
    additional_args: dict | None = None,
) -> dict:
    """
    A generic function to ask the LLM model a question and return
    the response in JSON format.

    Args:
        prompt (str): The prompt to ask the LLM model
        system_message (str): The system message to ask the LLM model
    """
    llm_call_logger.debug(f"LLM input: 'model': {llm}, 'messages': {prompt}")
    sys_message = enforce_json_format(system_message)
    response = await acompletion(
        model=llm,
        temperature=temperature,
        messages=[
            {"content": sys_message, "role": "system"},
            {"content": prompt, "role": "user"},
        ],
        api_key=api_key,
        additional_args=additional_args,
    )

    cost = completion_cost(response)
    result = {
        "answer": json.loads(response.choices[0].message.content),
        "cost": cost,
    }

    return result
