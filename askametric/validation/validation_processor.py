# Set up imports
from typing import Any, Callable
from askametric.utils import _ask_llm_json
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

    def __init__(self, llm: str) -> None:
        """
        Init

        auth_token: the authentication token for the LLM
        llm: the LLM to be used
        """
        self.llm = llm
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
        self, question: str, llm_response: str, **kwargs: Any
    ) -> dict[str, Any]:
        relevancy_prompt = get_relevancy_prompt(
            question=question, llm_response=llm_response
        )
        relevancy_evaluation = await _ask_llm_json(
            system_message=self.grading_bot_prompt,
            prompt=relevancy_prompt,
            llm=self.llm,
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
        **kwargs: Any,
    ) -> dict[str, Any]:
        accuracy_prompt = get_accuracy_prompt(
            correct_answer=correct_answer, llm_response=llm_response
        )
        accuracy_evaluation = await _ask_llm_json(
            system_message=self.grading_bot_prompt, prompt=accuracy_prompt, llm=self.llm
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
        self, question: str, llm_response: str, instructions: str, **kwargs: Any
    ) -> dict[str, Any]:
        instructions_prompt = get_instructions_prompt(
            question=question, instructions=instructions, llm_response=llm_response
        )
        instructions_evaluation = await _ask_llm_json(
            system_message=self.grading_bot_prompt,
            prompt=instructions_prompt,
            llm=self.llm,
        )
        instructions_evaluation = instructions_evaluation["answer"]
        return {f"instructions_{k}": val for k, val in instructions_evaluation.items()}

    async def test_consistency(
        self, question: str, llm_response: str, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Check if the LLM response is in a consistent format with the question
        """
        consistency_prompt = get_consistency_prompt(
            question=question, llm_response=llm_response
        )
        consistency_evaluation = await _ask_llm_json(
            system_message=self.grading_bot_prompt,
            prompt=consistency_prompt,
            llm=self.llm,
        )
        consistency_evaluation = consistency_evaluation["answer"]
        return {f"consistency_{k}": val for k, val in consistency_evaluation.items()}

    async def test_guardrails(
        self, guardrails_status: dict, **kwargs: Any
    ) -> dict[str, Any]:
        if all([val._value_ == "Passed" for val in guardrails_status.values()]):
            return {
                "guardrails_score": 0.0,
                "guardrails_reason": "All guardrails passed when they should not have",
            }
        else:
            not_passed = [
                key
                for key, val in guardrails_status.items()
                if val._value_ not in ["Passed", "Did not run"]
            ]
            return {
                "guardrails_score": 1.0,
                "guardrails_reason": f"These guardrails did not pass: {not_passed}",
            }

    async def get_eval_results(
        self, groundtruth: dict, response_to_evaluate: dict, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Get validation results for input

        Args:
            groundtruth: the groundtruth data
            response_to_evaluate: the response to evaluate
        """
        tests_to_run = groundtruth["tests_to_run"]
        results = {}

        for test in tests_to_run:
            try:
                test_result = await self.allowed_tests[test](
                    **groundtruth, **response_to_evaluate
                )
                results.update(test_result)
            except KeyError:
                print(f"Test `{test}` is not implemented. Not running this test")
                continue

        return results
