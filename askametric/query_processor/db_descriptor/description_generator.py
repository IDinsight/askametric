from .descriptor_prompts import (
    generate_description_prompt,
    generate_suggested_questions_prompt,
)
from ...utils import ask_llm_json, setup_logger, get_log_level_from_str
from ..tools import get_tools, track_time
import json
from sqlalchemy.ext.asyncio import AsyncSession
from cachetools import TTLCache
from aiocache import cached

_db_descriptor_instance = None


class DatabaseDescriptor:
    """Generates helpful descriptions for the database"""

    def __init__(
        self, llm: str = "gpt-4o", temperature: float = 0.1, log_level: str = "INFO"
    ):
        """
        Initialize the DatabaseDescriptor class.

        Args:
            llm (str): The name of the LLM model to use. Defaults to "gpt-4o".
            temperature (float): The temperature to use when generating descriptions.
                Defaults to 0.1.
            log_level (str): The log level to use. Defaults to "INFO".
        """
        self.llm = llm
        self.temperature = temperature

        self.logger = setup_logger("db_descriptor", get_log_level_from_str(log_level))

        self.tools = get_tools()
        self._description_cache: TTLCache = TTLCache(maxsize=100, ttl=60 * 60 * 24)
        self._description_cache["db_description"] = {}
        self._description_cache["suggested_questions"] = {}

    @track_time(create_class_attr="timings")
    @cached(ttl=60 * 60 * 24)
    async def generate_db_description(
        self,
        asession: AsyncSession,
        metric_db_id: str,
        sys_message: str,
        table_description: str,
        column_description: str = "",
        api_key: str | None = None,
    ) -> str:
        """
        Generate a database description.

        Args:
            system_prompt: The system prompt that sets the context for the
                database description.
            table_description: The description of the table in the database.
            column_description: The description of the columns in the table.
                Defaults to None.
            api_key: (Optional) API key to use for the LLM call
        """
        if metric_db_id not in self._description_cache["db_description"]:
            tables_list = [row["name"] for row in json.loads(table_description)]
            db_schema = await self.tools.get_tables_schema(
                tables_list, asession, metric_db_id
            )

            system, prompt = generate_description_prompt(
                system_prompt=sys_message,
                tables_description=table_description,
                db_schema=db_schema,
                column_description=column_description,
            )
            generated_description = await ask_llm_json(
                prompt=prompt,
                system_message=system,
                llm=self.llm,
                temperature=self.temperature,
                api_key=api_key,
            )
            self.logger.debug(
                f"Generated description for {metric_db_id}: {generated_description}"
            )
            self._description_cache["db_description"][
                metric_db_id
            ] = generated_description

        return self._description_cache["db_description"][metric_db_id]["answer"][
            "db_description"
        ]

    @track_time(create_class_attr="timings")
    @cached(ttl=60 * 60 * 24)
    async def generate_suggested_questions(
        self,
        asession: AsyncSession,
        metric_db_id: str,
        sys_message: str,
        table_description: str,
        column_description: str | None = None,
        api_key: str | None = None,
    ) -> str:
        """
        Generate suggested questions based on the database description.

        Args:
            system_prompt: The system prompt that sets the context for the
                suggested questions.
            table_description: The description of the table in the database.
            column_description: The description of the columns in the table.
                Defaults to None.
            api_key: (Optional) API key to use for the LLM call
        """
        if metric_db_id not in self._description_cache["suggested_questions"]:
            tables_list = [row["name"] for row in json.loads(table_description)]
            db_schema = await self.tools.get_tables_schema(
                tables_list, asession, metric_db_id
            )

            system, prompt = generate_suggested_questions_prompt(
                system_prompt=sys_message,
                tables_description=table_description,
                db_schema=db_schema,
                column_description=column_description or "",
            )
            generated_questions = await ask_llm_json(
                prompt=prompt,
                system_message=system,
                llm=self.llm,
                temperature=self.temperature,
                api_key=api_key,
            )
            self.logger.debug(
                f"Generated questions for {metric_db_id}:{generated_questions}"
            )
            self._description_cache["suggested_questions"][
                metric_db_id
            ] = generated_questions

        return self._description_cache["suggested_questions"][metric_db_id]["answer"][
            "suggested_questions"
        ]


def get_db_descriptor(
    llm: str = "gpt-4o", temperature: float = 0.1
) -> DatabaseDescriptor:
    """
    Return the DatabaseDescriptor instance.

    Args:
        llm (str): The name of the LLM model to use. Defaults to "gpt-4o".
        temperature (float): The temperature to use when generating descriptions.
            Defaults to 0.1.
    """
    global _db_descriptor_instance
    if _db_descriptor_instance is None:
        _db_descriptor_instance = DatabaseDescriptor(llm=llm, temperature=temperature)
    return _db_descriptor_instance
