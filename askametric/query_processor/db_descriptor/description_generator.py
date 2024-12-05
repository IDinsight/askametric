from .descriptor_prompts import generate_description_prompt
from ...utils import _ask_llm_json, setup_logger, get_log_level_from_str
from ..tools import get_tools, track_time
import json
from sqlalchemy.ext.asyncio import AsyncSession
from cachetools import TTLCache
from aiocache import cached


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

    @track_time(create_class_attr="timings")
    @cached(ttl=60 * 60 * 24)
    async def generate_db_description(
        self,
        asession: AsyncSession,
        metric_db_id: str,
        sys_message: str,
        table_description: str,
        column_description: str = "",
    ) -> str:
        """
        Generate a database description.

        Args:
            system_prompt: The system prompt that sets the context for the
                database description.
            table_description: The description of the table in the database.
            column_description: The description of the columns in the table.
                Defaults to None.
        """
        if metric_db_id not in self._description_cache:
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
            generated_description = await _ask_llm_json(
                prompt=prompt,
                system_message=system,
                llm=self.llm,
                temperature=self.temperature,
            )
            self.logger.debug(
                f"Generated description for {metric_db_id}: {generated_description}"
            )
            self._description_cache[metric_db_id] = generated_description

        return self._description_cache[metric_db_id]["answer"]["db_description"]
