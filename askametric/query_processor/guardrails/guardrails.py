from enum import Enum

from ...utils import _ask_llm_json
from .guardrails_prompts import (
    create_relevance_prompt,
    create_safety_prompt,
    create_check_code_prompt,
)


class GuardRailsStatus(Enum):
    """Status of the guard rails."""

    DID_NOT_RUN = "Did not run"
    PASSED = "Passed"
    IRRELEVANT = "Query Irrelevant"
    UNSAFE = "Query unsafe"
    CONTAINS_CODE = "Query contains code"


class LLMGuardRails:
    """Provides Functionality to
    run guard rails on query processing
    pipeline."""

    def __init__(
        self,
        gurdrails_llm: str,
        sys_message: str,
    ) -> None:
        """Initialize the GuardRails class."""
        self.cost = 0.0
        self.guardrails_llm = gurdrails_llm
        self.system_message = sys_message
        self.temperature = 0.0
        self.guardrails_status = {
            "relevance": GuardRailsStatus.DID_NOT_RUN,
            "safety": GuardRailsStatus.DID_NOT_RUN,
            "contains_code": GuardRailsStatus.DID_NOT_RUN,
        }

        self.safety_response = ""
        self.relevance_response = ""
        self.code_response = ""

    async def check_code(
        self,
        query: str,
    ) -> dict:
        """
        Handle the code in the query.
        """
        prompt = create_check_code_prompt(query)

        code_response = await _ask_llm_json(
            prompt=prompt,
            system_message=self.system_message,
            llm=self.guardrails_llm,
            temperature=self.temperature,
        )
        self.code = code_response["answer"]["contains_code"] == "True"
        if self.code is True:
            self.code_response = code_response["answer"]["response"]
            self.guardrails_status["contains_code"] = GuardRailsStatus.CONTAINS_CODE
        else:
            self.guardrails_status["contains_code"] = GuardRailsStatus.PASSED

        self.cost += float(code_response["cost"])
        return code_response

    async def check_safety(
        self, query: str, language: str, script: str, context: str
    ) -> dict:
        """
        Handle the PII/DML/prompt injection in the query.
        """
        prompt = create_safety_prompt(query, language, script, context=context)

        safety_response = await _ask_llm_json(
            prompt=prompt,
            system_message=self.system_message,
            llm=self.guardrails_llm,
            temperature=self.temperature,
        )
        self.safe = safety_response["answer"]["safe"] == "True"
        if self.safe is False:
            self.safety_response = safety_response["answer"]["response"]
            self.guardrails_status["safety"] = GuardRailsStatus.UNSAFE
        else:
            self.guardrails_status["safety"] = GuardRailsStatus.PASSED

        self.cost += float(safety_response["cost"])
        return safety_response

    async def check_relevance(
        self,
        query: str,
        language: str,
        script: str,
        table_description: str,
        context: str = "",
    ) -> dict:
        """
        Handle the relevance of the query.
        """
        prompt = create_relevance_prompt(
            query,
            language,
            script,
            table_description=table_description,
            context=context,
        )
        relevance_response = await _ask_llm_json(
            prompt=prompt,
            system_message=self.system_message,
            llm=self.guardrails_llm,
            temperature=self.temperature,
        )
        self.relevant = relevance_response["answer"]["relevant"] == "True"
        if self.relevant is False:
            self.relevance_response = relevance_response["answer"]["response"]
            self.guardrails_status["relevance"] = GuardRailsStatus.IRRELEVANT
        else:
            self.guardrails_status["relevance"] = GuardRailsStatus.PASSED

        self.cost += float(relevance_response["cost"])
        return relevance_response
