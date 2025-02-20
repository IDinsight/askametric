# Set up imports
from typing import Any, Callable
import pandas as pd
import numpy as np
from askametric.utils import ask_llm_json
from .validation_prompts import (
    grading_bot_prompt,
    get_relevancy_prompt,
    get_accuracy_prompt,
    get_instructions_prompt,
    get_consistency_prompt,
)


class QueryEvaluator:
    """
    Class that houses a series of functions to evaluate the LLMs performance on
    a specific database
    """

    def __init__(self, llm: str, temperature: float = 0.0) -> None:
        """
        Init

        llm: the LLM to be used
        temperature: the temperature to use. Default is 0.0
        """
        self.llm = llm
        self.temperature = temperature
        self.grading_bot_prompt = grading_bot_prompt()
        self.allowed_tests: dict[str, Callable] = {
            "Relevancy": self.test_relevancy,
            "Accuracy": self.test_accuracy,
            "Schema": self.test_schema,
            "Guardrails": self.test_guardrails,
            "Consistency": self.test_consistency,
            "Instructions": self.test_instructions,
        }

    async def test_relevancy(
        self,
        question: str,
        llm_response: str,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        relevancy_prompt = get_relevancy_prompt(
            question=question, llm_response=llm_response
        )
        relevancy_evaluation = await ask_llm_json(
            system_message=self.grading_bot_prompt,
            prompt=relevancy_prompt,
            llm=self.llm,
            temperature=self.temperature,
            api_key=api_key,
        )
        relevancy_evaluation = relevancy_evaluation["answer"]

        return {f"relevancy_{k}": val for k, val in relevancy_evaluation.items()}

    async def test_accuracy(
        self,
        correct_answer: str,
        correct_script: str,
        correct_language: str,
        llm_response: str,
        llm_ided_script: str,
        llm_ided_language: str,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        accuracy_prompt = get_accuracy_prompt(
            correct_answer=correct_answer, llm_response=llm_response
        )
        accuracy_evaluation = await ask_llm_json(
            system_message=self.grading_bot_prompt,
            prompt=accuracy_prompt,
            llm=self.llm,
            api_key=api_key,
        )
        accuracy_evaluation = accuracy_evaluation["answer"]

        # Do other checks here
        accuracy_evaluation["is_correct_language"] = 0.0
        accuracy_evaluation["is_correct_script"] = 0.0
        if correct_script == llm_ided_script:
            accuracy_evaluation["is_correct_script"] = 1.0
        if correct_language == llm_ided_language:
            accuracy_evaluation["is_correct_language"] = 1.0

        return {f"accuracy_{k}": val for k, val in accuracy_evaluation.items()}

    async def test_schema(
        self,
        correct_best_tables: list,
        correct_best_columns: dict[str, list],
        llm_ided_best_tables: list,
        llm_ided_best_columns: dict[str, list],
        **kwargs: Any,
    ) -> dict[str, Any]:
        schema_evaluation = {}

        # Check if the best tables and columns are correct
        schema_evaluation["schema_has_best_tables"] = 0.0
        schema_evaluation["schema_has_best_columns"] = 0.0
        num_common_tables = 0
        # LLM tables and columns must be a superset of the correct tables
        # and columns
        for table in correct_best_tables:
            if table in llm_ided_best_tables:
                num_common_tables += 1
                num_correct_cols = 0
                for col in correct_best_columns[table]:
                    if col in llm_ided_best_columns[table]:
                        num_correct_cols += 1
                if num_correct_cols == len(correct_best_columns[table]):
                    schema_evaluation["schema_has_best_columns"] += 1.0 / len(
                        correct_best_tables
                    )

        if num_common_tables == len(correct_best_tables):
            schema_evaluation["schema_has_best_tables"] = 1.0

        return schema_evaluation

    async def test_instructions(
        self,
        question: str,
        llm_response: str,
        instructions: str,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        instructions_prompt = get_instructions_prompt(
            question=question, instructions=instructions, llm_response=llm_response
        )
        instructions_evaluation = await ask_llm_json(
            system_message=self.grading_bot_prompt,
            prompt=instructions_prompt,
            llm=self.llm,
            temperature=self.temperature,
            api_key=api_key,
        )
        instructions_evaluation = instructions_evaluation["answer"]
        return {f"instructions_{k}": val for k, val in instructions_evaluation.items()}

    async def test_consistency(
        self,
        question: str,
        llm_response: str,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Check if the LLM response is in a consistent format with the question
        """
        consistency_prompt = get_consistency_prompt(
            question=question, llm_response=llm_response
        )
        consistency_evaluation = await ask_llm_json(
            system_message=self.grading_bot_prompt,
            prompt=consistency_prompt,
            llm=self.llm,
            temperature=self.temperature,
            api_key=api_key,
        )
        consistency_evaluation = consistency_evaluation["answer"]
        return {f"consistency_{k}": val for k, val in consistency_evaluation.items()}

    async def test_guardrails(
        self, guardrails_status: dict, **kwargs: Any
    ) -> dict[str, Any]:
        if all([val == "Passed" for val in guardrails_status.values()]):
            return {
                "guardrails_score": 0.0,
                "guardrails_reason": "All guardrails passed when they should not have",
            }
        else:
            not_passed = [
                key
                for key, val in guardrails_status.items()
                if val not in ["Passed", "Did not run"]
            ]
            return {
                "guardrails_score": 1.0,
                "guardrails_reason": f"These guardrails did not pass: {not_passed}",
            }

    async def evaluate(
        self, groundtruth: dict, response_to_evaluate: dict, api_key: str | None = None
    ) -> dict[str, Any]:
        """
        Get validation results for input

        Args:
            groundtruth: the groundtruth data
            response_to_evaluate: the response to evaluate
            api_key: (Optional) API Key for LLM calls
        """
        tests_to_run = groundtruth["tests_to_run"]
        results = {}

        results.update(groundtruth)
        results.update(response_to_evaluate)
        for test in tests_to_run:
            try:
                test_result = await self.allowed_tests[test](
                    **groundtruth, **response_to_evaluate, api_key=api_key
                )
                results.update(test_result)
            except KeyError:
                print(f"Test `{test}` is not implemented. Not running this test")
                continue

        return results

    async def get_eval_results(
        self,
        groundtruth_data: list[dict],
        responses_to_evaluate: list[dict],
        instructions: str,
        api_key: str | None = None,
    ):
        """
        Get evaluation results for a list of responses

        Args:
            groundtruth_data: the dictionary of groundtruth data
            responses_to_evaluate: the dictionary of responses to evaluate
            instructions: the instructions to evaluate against
            api_key: (Optional) API key for LLM calls
        """
        eval_results = []
        for i, val_question in enumerate(groundtruth_data):
            val_question["instructions"] = instructions

            llm_response = responses_to_evaluate[i]
            result = await self.evaluate(val_question, llm_response, api_key=api_key)
            eval_results.append(result)
        return pd.DataFrame(eval_results)

    @staticmethod
    def summarize_results(eval_results: pd.DataFrame):
        """
        Summarize numerical evaluation results in percentages

        Args:
            eval_results: the evaluation results
        """
        return (
            eval_results.select_dtypes(include=[np.number]).apply(np.nanmean, axis=0)
            * 100
        )
