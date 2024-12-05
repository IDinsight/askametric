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
    python parsable json with key "db_description"
    """

    return system, prompt
