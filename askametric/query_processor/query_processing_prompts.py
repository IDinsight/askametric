# Prompts for the pipeline


def get_query_language_prompt(query_text: str) -> tuple[str, str]:
    """Create prompt to get the language of the query."""

    system_message = "You are a highly-skilled linguist and polyglot.\
              Identify the language of the user query."

    prompt = f"""
    What language is the question asked in? What script is it written in?

    Examples -
    1. "How many beds are there?". Here the language is "English" and
    the script is "Latin".

    2. "vahaan kitane bistar hain?". Here the language is "Hindi" and
    the script is "Latin".

    3. "वहाँ कितने बिस्तर हैं?". Here the language is "Hindi" and
    the script is "Devanagari".

    Here is a question from a user -
    ### Question
    <<<{query_text}>>>

    Take a deep breath and work through the problem step-by-step

    Only, reply in a python parsable json with key "language"
    value being the language and "script" value being the script.
    """
    return system_message, prompt


def translation_prompt(
    query_model: dict,
    original_query_language: str,
    original_query_script: str,
    translated_query_language: str,
    translated_query_script: str,
) -> tuple[str, str]:
    """Create prompt to translate the original query for pipeline"""

    system_message = "You are a highly-skilled linguist and polyglot.\
              Translate the user query from an original into another input language."

    prompt = f"""
    Here is a question from a user who needs some text translated from
    {original_query_language} into {translated_query_language}.

    ===== Question =====
    <<< {query_model["query_text"]} >>>

    ===== Metadata =====
    Here is useful metadata (might be empty if not available):
    <<< {query_model["query_metadata"]} >>>

    ===== Current language =====
    The user query is currently in the following language:
    <<< {original_query_language} >>>

    ===== Current language script =====
    The user query is currently using the following script:
    <<< {original_query_script} >>>

    ===== Translation language =====
    Translate the user query into the following language:
    <<< {translated_query_language} >>>

    ===== Translation script =====
    Translate the user query into the following script:
    <<< {translated_query_script} >>>

    Take a deep breath and translate the text as accurately as possible.

    Only, reply in a python parsable json with key "query_text"
    value being the translated user query and "query_metadata"
    value being a translation of any available metadata.
    """
    return system_message, prompt


def create_question_type_prompt(query_text: str, chat_history: list) -> tuple[str, str]:
    """Create prompt to identify the type of question."""
    sys_message = "You are a highly-skilled linguist.\
    Your job is to look at the chat history and use it to infer what type of question \
    the user is asking."

    prompt = f"""
    ===== Question =====
    <<< {query_text} >>>

    ===== Chat History =====
    Here is the chat history (might be empty if not available):
    <<< {chat_history} >>>

    ===== Question Type =====
    There are 3 types of questions:
    1. New Question: this is a question that introduces a new topic, and makes sense
        without reference to the chat history.
    2. Follow-up Question: this is a question that builds on the chat history,
        and seeks NEW information on previously discussed topics.
    3. Clarification Question: this is a question that seeks to clarify something
        that was mentioned in the chat history, and seeks the SAME information as
        previously provided.

    Which of these three types does the Question fall under?
    Respond with either 1, 2, or 3 based on the type of question.

    ==== Response format ====
    python parsable json with key "question_type".
    """
    return sys_message, prompt


def create_reframe_query_prompt(query_text: str, chat_history: list) -> tuple[str, str]:
    """Create prompt to reframe the query based on chat history."""
    sys_message = "You are a highly-skilled linguist.\
    Your job is to look at the chat history and use it to infer context accurately\
    to understand what information the user is asking for with the current question."

    prompt = f"""
    ===== Question =====
    <<< {query_text} >>>

    ===== Chat History =====
    Here is the chat history (might be empty if not available):
    <<< {chat_history} >>>

    ===== Reframe question =====
    Is the question clear and unambiguous?
    If yes, you can leave the question as is.
    If not, reframe the question using information from the chat history.
    REMEMBER:
    - Keep the question as close to the original as possible, and
    - Pay more attention to the first question in the chat history.
    - PAY ATTENTION to words such as "these", "here", "it", etc. in the Question --
    they refer to places or people in the chat history.

    ==== Response format ====
    python parsable json with key "reframed_query".
    """
    return sys_message, prompt


def create_best_tables_prompt(query_model: dict, table_description: str) -> str:
    """Create prompt for best tables question."""
    prompt = f"""
    ===== Question =====
    <<< {query_model["query_text"]} >>>

    ===== Metadata =====
    Here is useful metadata (might be empty if not available):
    <<< {query_model["query_metadata"]} >>>

    ==== Source =====
    Which of the following sources of information are you going to use to
    answer the question. Select all that are relevant:
    {table_description}

    ===== Answer Format =====
    python parsable json with key "response_sources" and the value is
    JUST a list of tables.

    ==== Remember ====
    The person who is asking the question does not know about different codes or ids.
    Identify all information we would need to answer their question in names
    and numbers in natural language.
    """

    return prompt


def create_best_columns_prompt(
    query_model: dict, relevant_schemas: str, columns_description: str
) -> str:
    """Create prompt for best columns question."""
    prompt = f"""
    Here is a query that needs to be answered by conducting
    data analysis on a database.

    ===== Question =====
    <<< {query_model["query_text"]} >>>

    ===== Metadata =====
    Here is useful metadata (might be empty if not available):
    <<< {query_model["query_metadata"]} >>>

    ===== Relevant Tables =====
    Here is the tables schema of the relevant tables

    <<< {relevant_schemas} >>>

    ===== Columns =====
    Here is the description of columns (Might be empty if not available)
    <<< {columns_description} >>>

    ==== Relevant Columns =====
    Based on the above schema, which columns should we use to answer the question?
    It is much better to select more columns than less because the cost of
    omissions is high.

    ==== Response format ====
    python parsable json where each table is a key and the value is a list of columns.
    """

    return prompt


def create_sql_generating_prompt(
    query_model: dict,
    db_type: str,
    relevant_schemas: str,
    top_k_common_values: dict[str, dict],
    columns_description: str,
    num_common_values: int,
    indicator_vars: list,
) -> str:
    """Create prompt for generating SQL query."""
    prompt = f"""
    ===== Question =====
    <<< {query_model["query_text"]} >>>

    ===== Metadata =====
    Here is useful metadata (might be empty if not available):
    <<< {query_model["query_metadata"]} >>>

    ===== Relevant Tables =====
    The query will run on a {db_type} database with the following schema:
    <<<{relevant_schemas}>>>

    ===== Relevant Columns =====
    Here is the description of columns (Might be empty if not available)
    <<< {columns_description} >>>

    ===== Most common values in potentially relevant columns =====
    Here are a list of variables and their top {num_common_values} values. If
    a variable is in this special list: {indicator_vars}, the list of their unique
    values is exhaustive.
    <<<{top_k_common_values}>>>


    ==== Instruction ====
    Given the above, generate a SQL query that will answer the user's query.

    Always create a SQL query that will run on the above schema for the
    following SQL database type: {db_type}.

    Always use the query metadata to construct the SQL query.

    Add a LIMIT 10 if the result set is expected to be unnecessarily
    large (like 100+ rows). Otherwise, ensure that the query is exhaustive.

    Even for questions like "Best" or "Highest", ensure that the query is
    not only LIMIT 1 but LIMIT with some margin to ensure there are no ties.

    For complex queries involving UNION and ORDER BY,
    use the following and replicate its structure EXACTLY.
    Example:
    For a query like "What are the best-performing and worst-performing districts on
    indicator A?":
    ```
    SELECT district_name, indicator_A FROM (
        SELECT district_name, indicator_A
        FROM table_name
        ORDER BY indicator_A DESC
        LIMIT 1
    ) AS top_districts
    UNION ALL
    SELECT district_name, indicator_A FROM (
        SELECT district_name, indicator_A
        FROM table_name
        ORDER BY indicator_A ASC
        LIMIT 1
    ) AS bottom_districts;
    ```

    ===== Answer Format =====
    python parsable json with the key being "sql" and value being the sql code.
    """

    return prompt


def create_final_answer_prompt(
    query_model: dict,
    final_sql_code_to_run: str,
    final_sql_response: list,
    language: str,
    script: str,
) -> str:
    """Create prompt for final answer."""
    prompt = f"""
    Here is a question from a user -
    ### Question
    <<< {query_model["query_text"]} >>>

    ===== Metadata =====
    Here is useful metadata (might be empty if not available):
    <<< {query_model["query_metadata"]} >>>

    Here is a SQL query generated to answer that question -
    <<<{final_sql_code_to_run}>>>

    Here is the response to the SQL query from the DB -
    <<<{final_sql_response}>>>

    ===== Instruction =====
    Use the above information to create a final response for the user's
    question. You CANNOT provide information that is not available in the database.

    Always construct an answer that is as specific to the user as possible. Use the
    query metadata to do this.

    Use ALL the information in the response to the SQL query to answer accurately
    while also explaining how the answer was generated to the person who asked the
    question. Take care to reproduce decimals and fractions accurately.

    ===== Answer Format =====
    python parsable json with only one key "answer".

    Answer in {language} in the {script} script in the same
    mannerisms as the question.

    Remember, the user doesn't know what SQL is
    but are roughly familiar with what data is being
    collected a high level.
    """
    return prompt


def create_clarifying_answer_prompt(
    query_model: dict, chat_history: list, language: str, script: str
) -> str:
    """Create prompt to clarify the answer."""

    prompt = f"""
    ===== Question =====
    <<< {query_model["query_text"]} >>>

    ===== Metadata =====
    Here is useful metadata (might be empty if not available):
    <<< {query_model["query_metadata"]} >>>

    ===== Chat History =====
    <<< {chat_history} >>>

    ===== Clarifying Answer =====
    Based on the chat history, construct a final answer to the user's question.
    Use the chat history to clarify the answer as much as possible.

    Always construct an answer that is as specific to the user as possible. Use the
    query metadata to do this.

    Use ALL the information in the response to the SQL query to answer accurately
    while also explaining how the answer was generated to the person who asked the
    question. Take care to reproduce decimals and fractions accurately.

    ===== Answer Format =====
    python parsable json with only one key "answer".

    Answer in {language} in the {script} script in the same
    mannerisms as the question.

    Remember, the user doesn't know what SQL is
    but are roughly familiar with what data is being
    collected a high level.
    """
    return prompt
