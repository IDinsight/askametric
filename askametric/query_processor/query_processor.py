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
from .schemas import Prompts, UserQueryResponse


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
        self.guardrails_llm = guardrails_llm
        self.system_message = sys_message
        self.table_description = db_description
        self.column_description = column_description
        self.indicator_vars = indicator_vars
        self.num_common_values = num_common_values
        self.context_length = context_length

        self.context: list[dict] = []
        self.context_keys = ["text_response", "sql_query"]

    def _update_results(self) -> None:
        """
        Update the user query response with the results.
        """
        self.user_query_response.prompts = self.prompts
        self.user_query_response.processing_cost = self.cost
        self.user_query_response.guardrails_cost = self.guardrails.cost
        self.user_query_response.total_cost = self.cost + self.guardrails.cost
        self.user_query_response.guardrails_status = self.guardrails.guardrails_status

    def _update_context(self) -> None:
        """
        Update the context with the results.
        """
        context_dict = {"query": self.query}
        for k in self.context_keys:
            context_dict[k] = getattr(self.user_query_response, k)
        self.context.append(context_dict)
        self.context = self.context[-self.context_length :]

    @track_time(create_class_attr="timings")
    async def _get_query_language_from_llm(self) -> None:
        """
        The function asks the LLM model to identify the language
        of the user's query.
        """
        system_message, prompt = get_query_language_prompt(self.query["query_text"])

        query_language_llm_response = await _ask_llm_json(
            prompt=prompt,
            system_message=system_message,
            llm=self.llm,
            temperature=self.temperature,
        )

        self.prompts.language_prompt = prompt
        self.user_query_response.query_language = query_language_llm_response["answer"][
            "language"
        ]
        self.user_query_response.query_script = query_language_llm_response["answer"][
            "script"
        ]

        self.cost += float(query_language_llm_response["cost"])

    @track_time(create_class_attr="timings")
    async def _english_translation(self) -> dict:
        """
        The function asks the LLM model to translate the user query into English
        """
        system_message, prompt = english_translation_prompt(
            query_model=self.query,
            query_language=self.user_query_response.query_language,
            query_script=self.user_query_response.query_script,
        )

        eng_translation_llm_response = await _ask_llm_json(
            prompt=prompt,
            system_message=system_message,
            llm=self.llm,
            temperature=self.temperature,
        )

        eng_translation = eng_translation_llm_response["answer"]
        self.user_query_response.eng_translation = eng_translation["query_text"]

        self.cost += float(eng_translation_llm_response["cost"])
        return eng_translation

    @track_time(create_class_attr="timings")
    async def _get_conversation_summary_from_llm(self) -> None:
        """
        The function asks the LLM model to summarize the context
        of the previous conversation.
        """
        if len(self.context) > 0:
            prompt = create_conversation_summary_prompt(
                query_model=self.eng_translation,
                context=self.context,
                context_length=self.context_length,
            )
            convo_summary_llm_response = await _ask_llm_json(
                prompt=prompt,
                system_message=self.system_message,
                llm=self.llm,
                temperature=self.temperature,
            )
            self.user_query_response.conversation_summary = convo_summary_llm_response[
                "answer"
            ]["conversation_summary"]
            self.user_query_response.updated_query_text = convo_summary_llm_response[
                "answer"
            ]["updated_query_text"]
            self.user_query_response.text_response = convo_summary_llm_response[
                "answer"
            ]["final_answer"]

            self.cost += float(convo_summary_llm_response["cost"])
        else:
            prompt = ""
            self.user_query_response.conversation_summary = (
                "This is the beginning of the conversation."
            )
            self.user_query_response.updated_query_text = self.eng_translation[
                "query_text"
            ]
            self.user_query_response.text_response = ""

        self.prompts.conversation_summary_prompt = prompt

    @track_time(create_class_attr="timings")
    async def _get_best_tables_from_llm(self) -> None:
        """
        The function asks the LLM model to identify the best
        tables to answer a question.
        """
        prompt = create_best_tables_prompt(self.updated_query, self.table_description)
        best_tables_llm_response = await _ask_llm_json(
            prompt=prompt,
            system_message=self.system_message,
            context_message=self.user_query_response.conversation_summary,
            llm=self.llm,
            temperature=self.temperature,
        )

        self.prompts.best_tables_prompt = prompt
        self.user_query_response.best_tables = best_tables_llm_response["answer"][
            "response_sources"
        ]
        self.cost += float(best_tables_llm_response["cost"])

    @track_time(create_class_attr="timings")
    async def _get_best_columns_from_llm(self) -> None:
        """
        The function asks the LLM model to identify the best columns
        to answer a question.
        """
        self.user_query_response.relevant_schemas = await self.tools.get_tables_schema(
            self.user_query_response.best_tables,
            self.asession,
            metric_db_id=self.metric_db_id,
        )

        prompt = create_best_columns_prompt(
            self.updated_query,
            self.user_query_response.relevant_schemas,
            columns_description=self.column_description,
        )
        best_columns_llm_response = await _ask_llm_json(
            prompt=prompt,
            system_message=self.system_message,
            context_message=self.user_query_response.conversation_summary,
            llm=self.llm,
            temperature=self.temperature,
        )

        self.prompts.best_columns_prompt = prompt
        self.user_query_response.best_columns = best_columns_llm_response["answer"]
        self.cost += float(best_columns_llm_response["cost"])

    @track_time(create_class_attr="timings")
    async def _get_sql_query_from_llm(self) -> None:
        """
        The function asks the LLM model to generate a SQL query to
        answer the user's question.
        """
        top_k_common_values = await self.tools.get_common_column_values(
            table_column_dict=self.user_query_response.best_columns,
            asession=self.asession,
            num_common_values=self.num_common_values,
            indicator_vars=self.indicator_vars,
        )
        prompt = create_sql_generating_prompt(
            self.updated_query,
            self.db_type,
            str(self.user_query_response.relevant_schemas),
            top_k_common_values,
            self.column_description,
            self.num_common_values,
            # Maybe want to restrict to where theres intersection with best columns
            self.indicator_vars,
        )
        sql_query_llm_response = await _ask_llm_json(
            prompt=prompt,
            system_message=self.system_message,
            context_message=self.user_query_response.conversation_summary,
            llm=self.llm,
            temperature=self.temperature,
        )

        self.prompts.prompt_to_generate_sql = prompt
        self.user_query_response.sql_query = sql_query_llm_response["answer"]["sql"]
        self.cost += float(sql_query_llm_response["cost"])

    @track_time(create_class_attr="timings")
    async def _get_final_answer_from_llm(self) -> None:
        """
        The function asks the LLM model to generate the final
        answer to the user's question.
        """
        sql_result = await self.tools.run_sql(
            self.user_query_response.sql_query, self.asession
        )
        prompt = create_final_answer_prompt(
            self.updated_query,
            self.user_query_response.sql_query,
            sql_result,
            self.user_query_response.query_language,
            self.user_query_response.query_script,
        )
        final_answer_llm_response = await _ask_llm_json(
            prompt=prompt,
            system_message=self.system_message,
            context_message=self.user_query_response.conversation_summary,
            llm=self.llm,
            temperature=self.temperature,
        )

        self.user_query_response.text_response = final_answer_llm_response["answer"][
            "answer"
        ]
        self.cost += float(final_answer_llm_response["cost"])

    @track_time(create_class_attr="timings")
    async def process_query(self, query: dict) -> None:
        """
        The function processes the user query and returns the final answer.

        Args:
            query: The user's query. It is a dictionary with the
                following keys: query_text, query_metadata.
        """
        # --- Create / overwrite class attributes ---
        self.query = query
        self.cost = 0.0
        self.user_query_response = UserQueryResponse()
        self.guardrails = LLMGuardRails(self.guardrails_llm, self.system_message)
        self.prompts = Prompts()

        # --- Begin processing the query ---
        # Get query language
        await self._get_query_language_from_llm()

        # Translate query into English to be used for LLM's processing steps
        self.eng_translation = await self._english_translation()

        # Check query for code
        await self.guardrails.check_code(query=self.eng_translation["query_text"])
        if self.guardrails.code is True:
            self.user_query_response.text_response = self.guardrails.code_response
            self._update_results()

            return None

        # Get context for previous queries
        await self._get_conversation_summary_from_llm()
        self.updated_query = {
            "query_text": self.user_query_response.updated_query_text,
            "query_metadata": self.eng_translation["query_metadata"],
        }

        # Check query safety <-- need to do this after getting context
        # because the context is required to asses safety and relevance
        await self.guardrails.check_safety(
            query=self.updated_query["query_text"],
            language=self.user_query_response.query_language,
            script=self.user_query_response.query_script,
            context=self.user_query_response.conversation_summary,
        )
        if self.guardrails.safe is False:
            self.user_query_response.text_response = self.guardrails.safety_response
            self._update_results()

            return None

        # Check answer relevance <-- need to do this after getting context
        # because the context is required to asses safety and relevance
        await self.guardrails.check_relevance(
            query=self.updated_query["query_text"],
            language=self.user_query_response.query_language,
            script=self.user_query_response.query_script,
            table_description=self.table_description,
            context=self.user_query_response.conversation_summary,
        )

        if self.guardrails.relevant is False:
            self.user_query_response.text_response = self.guardrails.relevance_response
            self._update_results()

            return None

        if self.user_query_response.text_response != "":
            self._update_results()

            return None

        # Get best tables
        await self._get_best_tables_from_llm()

        # Get best columns
        await self._get_best_columns_from_llm()

        # Get SQL query
        await self._get_sql_query_from_llm()

        # Get final answer
        await self._get_final_answer_from_llm()

        # --- Update user query response ---
        self._update_results()

        # --- Update context ---
        self._update_context()
