# Set up imports
from typing import Any
from askametric.utils import _ask_llm_json
from .validation_prompts import (
    grading_bot_prompt,
    get_relevancy_prompt,
    get_accuracy_prompt,
    get_instructions_prompt,
)


class QueryEvaluator:
    """
    Class that houses a series of functions to evaluate the LLMs performance on
    a specific database
    """

    def __init__(self, llm: str) -> None:
        """
        Init

        auth_token: the authentication token for the LLM
        llm: the LLM to be used
        """
        self.llm = llm
        self.grading_bot_prompt = grading_bot_prompt()

    async def test_relevancy(self, question: str, llm_response: str):
        relevancy_prompt = get_relevancy_prompt(
            question=question, llm_response=llm_response
        )
        relevancy_evaluation = await _ask_llm_json(
            system_message=self.grading_bot_prompt,
            prompt=relevancy_prompt,
            llm=self.llm,
        )["answer"]

        return {f"relevancy_{k}": val for k, val in relevancy_evaluation.items()}

    async def test_accuracy(
        self,
        correct_answer: str,
        correct_script: str,
        correct_language: str,
        correct_best_columns: dict[str, list],
        correct_best_tables: list[str],
        llm_response: str,
        llm_ided_script: str,
        llm_ided_language: str,
        llm_ided_best_tables: list[str],
        llm_ided_best_columns: dict[str, list],
    ):
        accuracy_prompt = get_accuracy_prompt(
            correct_answer=correct_answer, llm_response=llm_response
        )
        accuracy_evaluation = await _ask_llm_json(
            system_message=self.grading_bot_prompt, prompt=accuracy_prompt, llm=self.llm
        )["answer"]

        # Do other checks here
        accuracy_evaluation["is_correct_language"] = 0.0
        accuracy_evaluation["is_correct_script"] = 0.0
        if correct_script == llm_ided_script:
            accuracy_evaluation["is_same_script"] = 1.0
        if correct_language == llm_ided_language:
            accuracy_evaluation["is_same_language"] = 1.0

        # Check if the best tables and columns are correct
        accuracy_evaluation["has_best_tables"] = 0.0
        accuracy_evaluation["has_best_columns"] = 0.0
        num_common_tables = 0
        for table in correct_best_tables:
            if table in llm_ided_best_tables:
                num_common_tables += 1
                num_correct_cols = 0
                for col in correct_best_columns[table]:
                    if col in llm_ided_best_columns[table]:
                        num_correct_cols += 1
                if num_correct_cols == len(correct_best_columns[table]):
                    accuracy_evaluation["has_best_columns"] += 1.0 / len(
                        correct_best_tables
                    )

        if num_common_tables == len(correct_best_tables):
            accuracy_evaluation["has_best_tables"] = 1.0

        return {f"accuracy_{k}": val for k, val in accuracy_evaluation.items()}

    async def test_instructions(
        self, question: str, llm_response: str, instructions: str
    ):
        instructions_prompt = get_instructions_prompt(
            question=question, instructions=instructions, llm_response=llm_response
        )
        instructions_evaluation = await _ask_llm_json(
            system_message=self.grading_bot_prompt,
            prompt=instructions_prompt,
            llm=self.llm,
        )["answer"]
        return {f"instructions_{k}": val for k, val in instructions_evaluation.items()}

    async def test_guardrails(self, guardrails_status: dict):
        if all([val == "Passed" for val in guardrails_status.values()]):
            return {
                "guardrails_score": 0.0,
                "guardrails_reason": "All guardrails passed when they should not have",
            }
        else:
            not_passed = [
                val
                for val in guardrails_status.values()
                if val not in ["Passed", "Did not run"]
            ]
        return {
            "guardrails_score": 1.0,
            "guardrails_reason": f"The following guardrails did not pass: {not_passed}",
        }

    async def get_eval_results(
        self, groundtruth: dict, response_to_evaluate: dict
    ) -> dict[str, Any]:
        """
        Get validation results for input

        Args:
            groundtruth: the groundtruth data
            response_to_evaluate: the response to evaluate
        """
        tests_to_run = groundtruth["tests_to_run"]
        if "Relevancy" in tests_to_run:
            relevancy_results = await self.test_relevancy(
                question=groundtruth["question"],
                llm_response=response_to_evaluate["llm_response"],
            )
        if "Accuracy" in tests_to_run:
            accuracy_results = await self.test_accuracy(
                correct_answer=groundtruth["correct_answer"],
                correct_script=groundtruth["correct_script"],
                correct_language=groundtruth["correct_language"],
                correct_best_columns=groundtruth["correct_best_columns"],
                correct_best_tables=groundtruth["correct_best_tables"],
                llm_response=response_to_evaluate["llm_response"],
                llm_ided_script=response_to_evaluate["llm_ided_script"],
                llm_ided_language=response_to_evaluate["llm_ided_language"],
                llm_ided_best_tables=response_to_evaluate["llm_ided_best_tables"],
                llm_ided_best_columns=response_to_evaluate["llm_ided_best_columns"],
            )
        if "Instructions" in tests_to_run:
            instructions_results = await self.test_instructions(
                question=groundtruth["question"],
                llm_response=response_to_evaluate["llm_response"],
                instructions=groundtruth["instructions"],
            )
        if "Guardrails" in tests_to_run:
            guardrails_results = await self.test_guardrails(
                guardrails_status=response_to_evaluate["guardrails_status"]
            )

        return {
            **relevancy_results,
            **accuracy_results,
            **instructions_results,
            **guardrails_results,
        }
