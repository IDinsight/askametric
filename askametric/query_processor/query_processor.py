from sqlalchemy.ext.asyncio import AsyncSession

from ..utils import _ask_llm_json
from ..utils import track_time
from .guardrails.guardrails import LLMGuardRails
from .query_processing_prompts import (
    create_best_columns_prompt,
    create_best_tables_prompt,
    create_conversation_summary_prompt,
    create_final_answer_prompt,
    create_sql_generating_prompt,
    english_translation_prompt,
    get_query_language_prompt,
)
from .tools import SQLTools, get_tools


def get_result_dict() -> dict:
    """Return a dictionary with the structure of the result."""
    return {
        "prompts": {
            "language_prompt": "",
            "best_tables_prompt": "",
            "best_columns_prompt": "",
            "sql_prompt": "",
            "final_answer_prompt": "",
            "conversation_summary_prompt": "",
            "prompt_to_generate_sql": "",
        },
        "text_response": "",
        "query_language": "",
        "query_script": "",
        "eng_translation": "",
        "best_tables": [],
        "best_columns": [],
        "schema_used": {},
        "sql_query": "",
        "total_cost": 0.0,
        "processing_cost": 0.0,
        "guardrails_cost": 0.0,
        "guardrails_status": {},
        "model_used": "",
        "guardrails_model": "",
        "conversation_summary": "",
    }


class LLMQueryProcessor:
    """Processes the user query and returns the final answer."""

    def __init__(
        self,
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
        context_length: int,
    ) -> None:
        """
        Initialize the QueryProcessor class.

        Args:
            asession: The SQLAlchemy AsyncSession object.
            metric_db_id: The database id to query.
            llm: The LLM model to use.
            guardrails_llm: The guardrails LLM model to use.
            sys_message: The system message to use.
            db_description: The description of the database.
            column_description: The description of the columns.
            indicator_vars: The indicator variables.
            num_common_values: The number of common values to get.
            context_length: The number of previous queries to consider.
        """
        self.asession = asession
        self.metric_db_id = metric_db_id
        self.db_type = db_type
        self.tools: SQLTools = get_tools()
        self.temperature: float = 0.1
        self.llm = llm
        self.system_message = sys_message
        self.table_description = db_description
        self.column_description = column_description
        self.indicator_vars = indicator_vars
        self.num_common_values = num_common_values
        self.context_length = context_length

        self.guardrails: LLMGuardRails = LLMGuardRails(
            gurdrails_llm=guardrails_llm, sys_message=self.system_message
        )
        self.cost: float = 0.0
        self.context: list[dict] = []
        self.context_keys = ["query", "text_response", "sql_query", "relevant_schemas"]

    @track_time(create_class_attr="timings")
    async def _get_query_language_from_llm(self, query: dict) -> tuple:
        """
        The function asks the LLM model to identify the language
        of the user's query.

        Args:
            query: The user's query. It is a dictionary with the
                following keys: query_text, query_metadata.
        """
        system_message, prompt = get_query_language_prompt(query["query_text"])
        language_prompt = prompt

        query_language_llm_response = await _ask_llm_json(
            prompt=prompt,
            system_message=system_message,
            llm=self.llm,
            temperature=self.temperature,
        )
        query_language = query_language_llm_response["answer"]["language"]
        query_script = query_language_llm_response["answer"]["script"]

        self.cost += float(query_language_llm_response["cost"])
        return language_prompt, query_language, query_script

    @track_time(create_class_attr="timings")
    async def _english_translation(
        self, query: dict, query_language: str, query_script: str
    ) -> None:
        """
        The function asks the LLM model to translate the user query into English

        Args:
            query: The user's query. It is a dictionary with the
                following keys: query_text, query_metadata.
            query_language: The language of the user's query.
            query_script: The script of the user's query.
        """
        system_message, prompt = english_translation_prompt(
            query_model=query,
            query_language=query_language,
            query_script=query_script,
        )

        eng_translation_llm_response = await _ask_llm_json(
            prompt=prompt,
            system_message=system_message,
            llm=self.llm,
            temperature=self.temperature,
        )

        eng_translation = eng_translation_llm_response["answer"]

        self.cost += float(eng_translation_llm_response["cost"])
        return eng_translation

    @track_time(create_class_attr="timings")
    async def _get_conversation_summary_from_llm(self, query: dict) -> tuple:
        """
        The function asks the LLM model to summarize the context
        of the previous conversation.
        """
        prompt = create_conversation_summary_prompt(
            query_model=query, context=self.context, context_length=self.context_length
        )
        convo_summary_llm_response = await _ask_llm_json(
            prompt=prompt,
            system_message=self.system_message,
            llm=self.llm,
            temperature=self.temperature,
        )
        convo_summary = convo_summary_llm_response["answer"]["conversation_summary"]
        final_answer = convo_summary_llm_response["answer"]["final_answer"]
        self.cost += float(convo_summary_llm_response["cost"])

        return prompt, convo_summary, final_answer

    @track_time(create_class_attr="timings")
    async def _get_best_tables_from_llm(self, query: dict, context: str = "") -> tuple:
        """
        The function asks the LLM model to identify the best
        tables to answer a question.

        Args:
            query: The user's query. It is a dictionary with the
                following keys: query_text, query_metadata.
            context: The context of the previous queries.
        """
        prompt = create_best_tables_prompt(query, self.table_description)
        best_tables_llm_response = await _ask_llm_json(
            prompt=prompt,
            system_message=self.system_message,
            context_message=context,
            llm=self.llm,
            temperature=self.temperature,
        )

        best_tables = best_tables_llm_response["answer"]["response_sources"]
        self.cost += float(best_tables_llm_response["cost"])
        return prompt, best_tables

    @track_time(create_class_attr="timings")
    async def _get_best_columns_from_llm(
        self, query: dict, best_tables: list, context: str = ""
    ) -> tuple:
        """
        The function asks the LLM model to identify the best columns
        to answer a question.

        Args:
            query: The user's query. It is a dictionary with the
                following keys: query_text, query_metadata.
            best_tables: The best tables identified to answer the question.
            context: The context of the previous queries.
        """
        relevant_schemas = await self.tools.get_tables_schema(
            best_tables, self.asession, metric_db_id=self.metric_db_id
        )
        prompt = create_best_columns_prompt(
            query,
            relevant_schemas,
            columns_description=self.column_description,
        )
        best_columns_llm_response = await _ask_llm_json(
            prompt=prompt,
            system_message=self.system_message,
            context_message=context,
            llm=self.llm,
            temperature=self.temperature,
        )

        best_columns = best_columns_llm_response["answer"]
        self.cost += float(best_columns_llm_response["cost"])
        return prompt, relevant_schemas, best_columns

    @track_time(create_class_attr="timings")
    async def _get_sql_query_from_llm(
        self,
        query: dict,
        best_columns: dict,
        relevant_schemas: dict,
        context: str = "",
    ) -> tuple:
        """
        The function asks the LLM model to generate a SQL query to
        answer the user's question.

        Args:
            query: The user's query. It is a dictionary with the
                following keys: query_text, query_metadata.
            best_columns: The best columns identified to answer the question.
            relevant_schemas: The relevant schemas of the tables.
            context: The context of the previous queries.
        """
        top_k_common_values = await self.tools.get_common_column_values(
            table_column_dict=best_columns,
            asession=self.asession,
            num_common_values=self.num_common_values,
            indicator_vars=self.indicator_vars,
        )
        prompt = create_sql_generating_prompt(
            query,
            self.db_type,
            str(relevant_schemas),
            top_k_common_values,
            self.column_description,
            self.num_common_values,
            # Maybe want to restrict to where theres intersection with best columns
            self.indicator_vars,
        )
        sql_query_llm_response = await _ask_llm_json(
            prompt=prompt,
            system_message=self.system_message,
            context_message=context,
            llm=self.llm,
            temperature=self.temperature,
        )

        sql_query = sql_query_llm_response["answer"]["sql"]
        self.cost += float(sql_query_llm_response["cost"])

        return prompt, sql_query

    @track_time(create_class_attr="timings")
    async def _get_final_answer_from_llm(
        self,
        query: dict,
        sql_query: str,
        query_language: str,
        query_script: str,
        context: str = "",
    ) -> tuple:
        """
        The function asks the LLM model to generate the final
        answer to the user's question.

        Args:
            query: The user's query. It is a dictionary with the
                following keys: query_text, query_metadata.
            sql_query: The SQL query generated to answer the user's question.
            query_language: The language of the user's query.
            query_script: The script of the user's query.
            context: The context of the previous queries.
        """
        sql_result = await self.tools.run_sql(sql_query, self.asession)
        prompt = create_final_answer_prompt(
            query,
            sql_query,
            sql_result,
            query_language,
            query_script,
        )
        final_answer_llm_response = await _ask_llm_json(
            prompt=prompt,
            system_message=self.system_message,
            context_message=context,
            llm=self.llm,
            temperature=self.temperature,
        )

        final_answer = final_answer_llm_response["answer"]["answer"]
        self.cost += float(final_answer_llm_response["cost"])

        return prompt, sql_result, final_answer

    @track_time(create_class_attr="timings")
    async def process_query(self, query: dict) -> dict:
        """
        The function processes the user query and returns the final answer.

        Args:
            query: The user's query. It is a dictionary with the
                following keys: query_text, query_metadata.
        """
        result_dict = get_result_dict()

        result_dict["query"] = query

        # Get query language
        (
            result_dict["prompts"]["language_prompt"],
            result_dict["query_language"],
            result_dict["query_script"],
        ) = await self._get_query_language_from_llm(query=query)

        # Translate query into English to be used for LLM's processing steps
        eng_translation = await self._english_translation(
            query=query,
            query_language=result_dict["query_language"],
            query_script=result_dict["query_script"],
        )
        result_dict["eng_translation"] = eng_translation["query_text"]

        # Check query for code
        code_response = await self.guardrails.check_code(
            query=eng_translation["query_text"]
        )
        if self.guardrails.code is True:
            result_dict["text_response"] = code_response
            result_dict["processing_cost"] = self.cost

            self.cost = 0.0

            return result_dict

        # Get context for previous queries
        (
            result_dict["prompts"]["conversation_summary_prompt"],
            result_dict["conversation_summary"],
            result_dict["text_response"],
        ) = await self._get_conversation_summary_from_llm(query=query)

        if result_dict["text_response"] != "":
            result_dict["processing_cost"] = self.cost

            self.cost = 0.0

            return result_dict

        # Check query safety
        await self.guardrails.check_safety(
            query=eng_translation["query_text"],
            language=result_dict["query_language"],
            script=result_dict["query_script"],
            context=result_dict["conversation_summary"],
        )
        if self.guardrails.safe is False:
            result_dict["text_response"] = self.guardrails.safety_response
            result_dict["processing_cost"] = self.cost

            self.cost = 0.0

            return result_dict

        # Check answer relevance
        await self.guardrails.check_relevance(
            query=eng_translation["query_text"],
            language=result_dict["query_language"],
            script=result_dict["query_script"],
            table_description=self.table_description,
            context=result_dict["conversation_summary"],
        )

        if self.guardrails.relevant is False:
            result_dict["text_response"] = self.guardrails.relevance_response
            result_dict["processing_cost"] = self.cost

            self.cost = 0.0

            return result_dict

        # Get best tables
        (
            result_dict["prompts"]["best_tables_prompt"],
            result_dict["best_tables"],
        ) = await self._get_best_tables_from_llm(
            query=eng_translation,
            context=result_dict["conversation_summary"],
        )

        # Get best columns
        (
            result_dict["prompts"]["best_columns_prompt"],
            result_dict["relevant_schemas"],
            result_dict["best_columns"],
        ) = await self._get_best_columns_from_llm(
            query=eng_translation,
            best_tables=result_dict["best_tables"],
            context=result_dict["conversation_summary"],
        )

        # Get SQL query
        (
            result_dict["prompts"]["top_k_column_values_prompt"],
            result_dict["sql_query"],
        ) = await self._get_sql_query_from_llm(
            query=eng_translation,
            best_columns=result_dict["best_columns"],
            relevant_schemas=result_dict["relevant_schemas"],
            context=result_dict["conversation_summary"],
        )

        # Get final answer
        (
            result_dict["prompts"]["final_answer_prompt"],
            result_dict["sql_result"],
            result_dict["text_response"],
        ) = await self._get_final_answer_from_llm(
            query=eng_translation,
            sql_query=result_dict["sql_query"],
            query_language=result_dict["query_language"],
            query_script=result_dict["query_script"],
            context=result_dict["conversation_summary"],
        )

        result_dict["cost"] = self.cost

        self.context.append({k: result_dict[k] for k in self.context_keys})

        self.context = self.context[-self.context_length :]
        self.cost = 0.0

        return result_dict
