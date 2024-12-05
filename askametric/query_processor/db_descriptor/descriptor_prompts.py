def generate_description_prompt(
    system_prompt: str,
    tables_description: str,
    db_schema: str,
    column_description: str = "",
) -> tuple[str, str]:
    """
    Create prompt to get DB description
    """
    system = """
    You are an expert in semantics and contextualization
    Provide a DESCRIPTION the information available in the database,
    do NOT provide specific numbers.
    Be clear and descriptive, but concise.
    Use bullet points and lists, to make for easier reading.
    """
    prompt = f"""
    ===== User input description =====
    {system_prompt}

    ===== Database tables =====
    {tables_description}

    ===== Database columns (may be empty)=====
    {column_description}

    ===== Database schema =====
    {db_schema}


    ==== Database Description ====
    Based on the user input description, tables, columns and schema, produce a SUMMARY
    that answers the following questions:
    What type of information is present in the database?
    What information is NOT in the database?

    ==== Answer Format ====
    python parsable json with key "db_description" and a string as the description.
    """

    return system, prompt


def generate_suggested_questions_prompt(
    system_prompt: str,
    tables_description: str,
    db_schema: str,
    column_description: str = "",
    chat_history: list = [],
) -> tuple[str, str]:
    """
    Create prompt to get suggested questions
    """
    system = """
    You are a technical advisor to non-technical decision-makers.
    Your job is to suggest SIMPLE questions these officials can ask based on the data
    available in the database, to access information that can help them make better
    decisions.
    These questions MUST be answerable using the information available in the database.
    """
    prompt = f"""
    ===== User input description =====
    {system_prompt}

    ===== Database tables =====
    {tables_description}

    ===== Database columns (may be empty)=====
    {column_description}

    ===== Database schema =====
    {db_schema}

    ===== Chat history (may be empty)=====
    {chat_history}


    ==== Suggested Questions ====
    Based on the user input description, tables, columns, schema, and chat history,
    provide a list of 5 NEW QUESTIONS that can be answered using the information
    available in the database.
    These questions should be SIMPLE, and NOT ask for correlation, relationship, etc.
    Direct comparisons between datapoints or aggregated data are allowed.

    ==== Answer Format ====
    python parsable json with key "suggested_questions" and list of questions
    as a string.
    """

    return system, prompt
