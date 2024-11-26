import ast
import asyncio
import glob
import json
import os
import time
from datetime import datetime
from typing import Dict, List

import dotenv
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from askametric.query_processor.query_processor import LLMQueryProcessor
from askametric.validation.validation_processor import QueryEvaluator

dotenv.load_dotenv()


async def main(args):
    if args.path_to_data_sources:
        data_source_files = glob.glob(f"{args.path_to_data_sources}/*.sqlite")
    else:
        data_source_files = glob.glob(f"{DATA_SOURCES_PATH}/*.sqlite")

    if not data_source_files:
        raise FileNotFoundError(
            f"No .sqlite files found in "
            f"{args.path_to_data_sources or DATA_SOURCES_PATH}"
        )

    all_results_df = pd.DataFrame()
    evaluator = QueryEvaluator(llm=VAL_LLM)

    for i, data_source_file in enumerate(data_source_files):
        db_name = os.path.splitext(os.path.basename(data_source_file))[0]
        print(f"Processing {db_name}... ({i+1} of {len(data_source_files)})")

        test_cases = get_test_cases(db_name)
        env_vars = get_env_vars(db_name)

        results_df = await process_queries(
            db_name=db_name, test_cases=test_cases, env_vars=env_vars
        )

        print(f"Starting evaluation for {db_name}...")
        eval_results_df = await evaluate_results(
            evaluator=evaluator,
            llm_responses=results_df,
            test_cases=test_cases,
            sys_message=env_vars["system_message"],
        )

        all_results_df = pd.concat([all_results_df, eval_results_df])

    # Remove random unnamed cols
    all_results_df = all_results_df.loc[
        :, ~all_results_df.columns.str.contains("^Unnamed")
    ]

    # Save results
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results_df.to_csv(f"{RESULTS_PATH}/results_{date}.csv", index=False)
    average_accuracy = round(
        (
            all_results_df["accuracy_score"].sum(skipna=True)
            + all_results_df["guardrails_score"].sum(skipna=True)
        )
        / len(all_results_df),
        2,
    )
    print(f"\n\nAverage accuracy: {average_accuracy}")
    print(f"\n\n To see more, look at validation/{RESULTS_PATH}/results.csv")


def get_env_vars(db_name: str) -> Dict:
    env_vars = {}
    for var in [
        "SYSTEM_MESSAGE",
        "DB_TABLE_DESCRIPTION",
        "DB_COLUMN_DESCRIPTION",
        "INDICATOR_VARS",
        "NUM_COMMON_VALUES",
    ]:
        key = f"{db_name.upper()}_{var}"
        value = os.getenv(key)
        if value is None:
            print(f"Warning: {key} is None")
        env_vars[var.lower()] = value
    return env_vars


def get_test_cases(db_name: str) -> pd.DataFrame:
    return pd.read_csv(f"{TEST_CASES_PATH}/{db_name}.csv")


async def process_single_query(
    db_name: str, query_data: Dict, async_session, env_vars: Dict
) -> Dict:
    try:
        async with async_session() as asession:
            query_processor = LLMQueryProcessor(
                query={
                    "query_text": query_data["question"],
                    "query_metadata": query_data["question_metadata"],
                },
                asession=asession,
                metric_db_id=db_name,
                db_type="sqlite",
                llm=LLM,
                guardrails_llm=GUARDRAILS_LLM,
                sys_message=env_vars["system_message"],
                db_description=env_vars["db_table_description"],
                column_description=env_vars["db_column_description"],
                indicator_vars=env_vars["indicator_vars"],
                num_common_values=env_vars["num_common_values"],
                log_level=LOG_LEVEL,
            )

            await query_processor.process_query()

            return {
                "db_name": db_name,
                "llm_response": query_processor.final_answer,
                "llm_ided_script": query_processor.query_script,
                "llm_ided_language": query_processor.query_language,
                "guardrails_status": {
                    k: v.value
                    for k, v in query_processor.guardrails.guardrails_status.items()
                },
                "llm_ided_best_tables": query_processor.best_tables,
                "llm_ided_best_columns": query_processor.best_columns,
                "request_status": "Success",
            }
    except Exception as e:
        print(f"Error processing query: {e}")
        return {
            "db_name": db_name,
            "llm_response": None,
            "llm_ided_script": None,
            "llm_ided_language": None,
            "guardrails_status": None,
            "llm_ided_best_tables": None,
            "llm_ided_best_columns": None,
            "request_status": f"Error: {str(e)}",
        }


async def process_queries(
    db_name: str, test_cases: pd.DataFrame, env_vars: Dict
) -> pd.DataFrame:
    async_session = get_async_session(db_name)

    # Create batches of tasks
    tasks = []
    for _, test_case in test_cases.iterrows():
        task = process_single_query(
            db_name=db_name,
            query_data=test_case.to_dict(),
            async_session=async_session,
            env_vars=env_vars,
        )
        tasks.append(task)

    # Process tasks in batches
    results = []
    for i in range(0, len(tasks), MAX_CONCURRENT_TASKS):
        batch = tasks[i : i + MAX_CONCURRENT_TASKS]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
        print(f"LLM Responses {i + len(batch)}/{len(tasks)} queries for {db_name}")

    return pd.DataFrame(results)


async def evaluate_single_result(
    evaluator: QueryEvaluator,
    result: Dict,
    groundtruth: Dict,
    sys_message: str,
) -> pd.DataFrame:
    """Evaluate a single result"""
    try:
        eval_result = await evaluator.get_eval_results(
            groundtruth_data=[groundtruth],
            responses_to_evaluate=[result],
            instructions=sys_message,
        )
        eval_result["db_name"] = result["db_name"]
        return eval_result
    except Exception as e:
        print(f"Error evaluating result: {e}")
        return pd.DataFrame(
            [
                {
                    "db_name": result["db_name"],
                    "question": groundtruth["question"],
                    "question_metadata": groundtruth["question_metadata"],
                    "eval_status": f"Error: {str(e)}",
                    "request_status": result["request_status"],
                }
            ]
        )


async def evaluate_results(
    evaluator: QueryEvaluator,
    llm_responses: pd.DataFrame,
    test_cases: pd.DataFrame,
    sys_message: str,
) -> pd.DataFrame:
    groundtruth_data = reformat_input_data(test_cases)
    responses_to_evaluate = llm_responses.to_dict(orient="records")

    evaluation_tasks = [
        evaluate_single_result(
            evaluator=evaluator,
            result=result,
            groundtruth=groundtruth_data[i],
            sys_message=sys_message,
        )
        for i, result in enumerate(responses_to_evaluate)
    ]

    all_eval_results = await asyncio.gather(*evaluation_tasks)

    return pd.concat(all_eval_results, ignore_index=True)


def _convert_columns(df: pd.DataFrame, column_name: str, type: str) -> pd.DataFrame:
    if type == "list":
        df[column_name] = df[column_name].apply(lambda x: ast.literal_eval(x))
    elif type == "dict":
        df[column_name] = df[column_name].apply(lambda x: x.replace("'", '"'))
        df[column_name] = df[column_name].apply(lambda x: json.loads(x))
    return df


def reformat_input_data(input_data: pd.DataFrame) -> List[Dict]:
    conversion_dict = {
        "correct_best_tables": "list",
        "tests_to_run": "list",
        "correct_best_columns": "dict",
        "question_metadata": "dict",
    }

    for column_name, type in conversion_dict.items():
        input_data = _convert_columns(input_data, column_name, type)

    return [row.to_dict() for _, row in input_data.iterrows()]


def get_async_session(db_name: str):
    aengine = create_async_engine(
        url=f"sqlite+aiosqlite:///{DATA_SOURCES_PATH}/{db_name}.sqlite",
    )
    async_session = sessionmaker(
        bind=aengine, class_=AsyncSession, expire_on_commit=False
    )
    return async_session


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_data_sources", type=str, default=None)
    args = parser.parse_args()

    DATA_SOURCES_PATH = args.path_to_data_sources or "data_sources"

    TEST_CASES_PATH = "test_cases"
    RESULTS_PATH = "results"
    MAX_CONCURRENT_TASKS = 20
    LLM = os.environ["LLM"]
    VAL_LLM = os.environ["VAL_LLM"]
    GUARDRAILS_LLM = os.environ["GUARDRAILS_LLM"]
    LOG_LEVEL = os.environ["LOG_LEVEL"]

    time_start = time.time()
    asyncio.run(main(args))

    print(f"Total time taken: {round(time.time() - time_start, 2)}s")
