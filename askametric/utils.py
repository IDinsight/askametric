import json
import logging
import time
from functools import wraps
from logging import Logger
from typing import Any, Callable

from aiocache import cached
from litellm import acompletion, completion_cost
from tenacity import retry, stop_after_attempt, stop_before_delay


def get_log_level_from_str(log_level_str: str = "INFO") -> int:
    """
    Get log level from string
    """
    log_level_dict = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }

    return log_level_dict.get(log_level_str.upper(), logging.INFO)


def setup_logger(
    name: str = __name__, log_level: int | str = get_log_level_from_str()
) -> Logger:
    """
    Setup logger for the application
    """
    logger = logging.getLogger(name)

    # If the logger already has handlers,
    # assume it was already configured and return it.
    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    formatter = logging.Formatter(
        "%(message)s - %(asctime)s - %(filename)20s:%(lineno)4s\n",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )

    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


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
    llm_config: dict | None = None,
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
        **llm_config if llm_config else {},
    )

    cost = completion_cost(response)
    result = {
        "answer": json.loads(response.choices[0].message.content),
        "cost": cost,
    }

    return result


def track_time(create_class_attr: str) -> Callable:
    """
    Decorator to add time tracking within classes.

    It adds an attribute "create_attr" to the class instance
    if it does not exist. Else, it appends the time taken
    by the function to the attribute.
    """

    def decorator(func: Callable) -> Callable:
        """Decorator"""

        @wraps(func)
        async def wrapper(self: Any, *args: str, **kwargs: str) -> Any:
            """Wrapper"""
            start_time = time.time()
            result = await func(self, *args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            if hasattr(self, create_class_attr):
                getattr(self, create_class_attr)[func.__name__] = elapsed_time
            else:
                setattr(self, create_class_attr, {func.__name__: elapsed_time})
            return result

        return wrapper

    return decorator
