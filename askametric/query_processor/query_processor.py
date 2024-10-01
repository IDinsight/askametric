from sqlalchemy.ext.asyncio import AsyncSession

from ..utils import _ask_llm_json
from ..utils import track_time
from .guardrails.guardrails import LLMGuardRails
from .query_processing_prompts import (
    create_best_columns_prompt,
    create_best_tables_prompt,
    create_final_answer_prompt,
    create_sql_generating_prompt,
    english_translation_prompt,
    get_query_language_prompt,
)
from .tools import SQLTools, get_tools


class LLMQueryProcessor:
    """Processes the user query and returns the final answer."""

    def __init__(
        self,
        query: dict,
        asession: AsyncSession,
        metric_db_id: str,
        db_type: str,
        llm: str,
        guardrails_llm: str,
        sys_message: str,
        db_description: str,
        column_description: str,
        indicator_vars: list,
        num_common_values: int,
    ) -> None:
        """
        Initialize the QueryProcessor class.

        Args:
            query: The user query and query metadata.
            asession: The SQLAlchemy AsyncSession object.
            metric_db_id: The database id to query.
            llm: The LLM model to use.
            guardrails_llm: The guardrails LLM model to use.
            sys_message: The system message to use.
            db_description: The description of the database.
            column_description: The description of the columns.
            indicator_vars: The indicator variables.
            num_common_values: The number of common values to get.
        """
        self.query = query
        self.asession = asession
        self.metric_db_id = metric_db_id
        self.db_type = db_type
        self.tools: SQLTools = get_tools()
        self.temperature = 0.1
        self.llm = llm
        self.system_message = sys_message
        self.table_description = db_description
        self.column_description = column_description
        self.indicator_vars = indicator_vars
        self.num_common_values = num_common_values
        self.guardrails: LLMGuardRails = LLMGuardRails(
            guardrails_llm, self.system_message
        )
        self.cost = 0.0
        self.language_prompt = ""
        self.query_language = ""
        self.script = ""
        self.eng_translation: dict = {}
        self.best_tables: list[str] = []
        self.best_columns: dict[str, list[str]] = {}
        self.top_k_common_values: dict[str, dict] = {}
        self.sql_query: str = ""
        self.final_answer: str = ""
        self.relevant_schemas: str = ""
        self.best_tables_prompt: str = ""
        self.best_columns_prompt: str = ""
        self.sql_generating_prompt: str = ""
        self.final_answer_prompt: str = ""

    @track_time(create_class_attr="timings")
    async def _get_query_language_from_llm(self) -> None:
        """
        The function asks the LLM model to identify the language
        of the user's query.
        """
        system_message, prompt = get_query_language_prompt(self.query["query_text"])
        self.language_prompt = prompt

        query_language_llm_response = await _ask_llm_json(
            prompt, system_message, llm=self.llm, temperature=self.temperature
        )
        self.query_language = query_language_llm_response["answer"]["language"]
        self.query_script = query_language_llm_response["answer"]["script"]

        self.cost += float(query_language_llm_response["cost"])

    @track_time(create_class_attr="timings")
    async def _english_translation(self) -> None:
        """
        The function asks the LLM model to translate the user query into English
        """
        system_message, prompt = english_translation_prompt(
            query_model=self.query,
            query_language=self.query_language,
            query_script=self.query_script,
        )

        eng_translation_llm_response = await _ask_llm_json(
            prompt, system_message, llm=self.llm, temperature=self.temperature
        )

        self.eng_translation = eng_translation_llm_response["answer"]

        self.cost += float(eng_translation_llm_response["cost"])

    @track_time(create_class_attr="timings")
    async def _get_best_tables_from_llm(self) -> None:
        """
        The function asks the LLM model to identify the best
        tables to answer a question.
        """
        prompt = create_best_tables_prompt(self.eng_translation, self.table_description)
        best_tables_llm_response = await _ask_llm_json(
            prompt, self.system_message, llm=self.llm, temperature=self.temperature
        )

        self.best_tables = best_tables_llm_response["answer"]["response_sources"]
        self.cost += float(best_tables_llm_response["cost"])
        self.best_tables_prompt = prompt

    @track_time(create_class_attr="timings")
    async def _get_best_columns_from_llm(self) -> None:
        """
        The function asks the LLM model to identify the best columns
        to answer a question.
        """
        self.relevant_schemas = await self.tools.get_tables_schema(
            self.best_tables, self.asession, metric_db_id=self.metric_db_id
        )
        prompt = create_best_columns_prompt(
            self.eng_translation,
            self.relevant_schemas,
            columns_description=self.column_description,
        )
        best_columns_llm_response = await _ask_llm_json(
            prompt, self.system_message, llm=self.llm, temperature=self.temperature
        )

        self.best_columns = best_columns_llm_response["answer"]
        self.cost += float(best_columns_llm_response["cost"])
        self.best_columns_prompt = prompt

    @track_time(create_class_attr="timings")
    async def _get_sql_query_from_llm(self) -> None:
        """
        The function asks the LLM model to generate a SQL query to
        answer the user's question.
        """
        self.top_k_common_values = await self.tools.get_common_column_values(
            table_column_dict=self.best_columns,
            asession=self.asession,
            num_common_values=self.num_common_values,
            indicator_vars=self.indicator_vars,
        )
        prompt = create_sql_generating_prompt(
            self.eng_translation,
            self.db_type,
            self.relevant_schemas,
            self.top_k_common_values,
            self.column_description,
            self.num_common_values,
            # Maybe want to restrict to where theres intersection with best columns
            self.indicator_vars,
        )
        sql_query_llm_response = await _ask_llm_json(
            prompt, self.system_message, llm=self.llm, temperature=self.temperature
        )

        self.sql_query = sql_query_llm_response["answer"]["sql"]
        self.cost += float(sql_query_llm_response["cost"])
        self.sql_generating_prompt = prompt

    @track_time(create_class_attr="timings")
    async def _get_final_answer_from_llm(self) -> None:
        """
        The function asks the LLM model to generate the final
        answer to the user's question.
        """
        sql_result = await self.tools.run_sql(self.sql_query, self.asession)
        prompt = create_final_answer_prompt(
            self.eng_translation,
            self.sql_query,
            sql_result,
            self.query_language,
            self.query_script,
        )
        final_answer_llm_response = await _ask_llm_json(
            prompt, self.system_message, llm=self.llm, temperature=self.temperature
        )

        self.final_answer = final_answer_llm_response["answer"]["answer"]
        self.cost += float(final_answer_llm_response["cost"])
        self.final_answer_prompt = prompt

    @track_time(create_class_attr="timings")
    async def process_query(self) -> None:
        """
        The function processes the user query and returns the final answer.
        """
        # Get query language
        await self._get_query_language_from_llm()

        # Translate query into English to be used for LLM's processing steps
        await self._english_translation()

        # Check query safety
        await self.guardrails.check_safety(
            self.eng_translation["query_text"], self.query_language, self.query_script
        )
        if self.guardrails.safe is False:
            self.final_answer = self.guardrails.safety_response
            return None

        # Check answer relevance
        await self.guardrails.check_relevance(
            self.eng_translation["query_text"],
            self.query_language,
            self.query_script,
            self.table_description,
        )

        if self.guardrails.relevant is False:
            self.final_answer = self.guardrails.relevance_response
            return None

        await self._get_best_tables_from_llm()
        await self._get_best_columns_from_llm()
        await self._get_sql_query_from_llm()
        await self._get_final_answer_from_llm()
