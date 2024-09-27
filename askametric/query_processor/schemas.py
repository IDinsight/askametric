from pydantic import BaseModel, ConfigDict, Field


class Prompts(BaseModel):
    """
    Pydantic model for prompts
    """

    final_answer_prompt: str | None = None
    best_columns_prompt: str | None = None
    best_tables_prompt: str | None = None
    language_prompt: str | None = None
    prompt_to_generate_sql: str | None = None
    conversation_summary_prompt: str | None = None

    model_config = ConfigDict(from_attributes=True)


class UserQueryResponse(BaseModel):
    """
    Pydantic model for response to Query
    """

    best_tables: list[str] | str = Field(default="")
    query_language: str = Field(default="")
    query_script: str = Field(default="")
    best_columns: dict | str = Field(default="")
    sql_query: str = Field(default="")
    text_response: str = Field(default="")
    total_cost: float = Field(default=0.0)
    processing_cost: float = Field(default=0.0)
    guardrails_cost: float = Field(default=0.0)
    guardrails_status: dict = Field(default={})
    relevant_schemas: str = Field(default="")
    prompts: Prompts = Field(default_factory=dict)
    eng_translation: str = Field(default="")
    updated_query_text: str = Field(default="")
    conversation_summary: str = Field(default="")

    model_config = ConfigDict(from_attributes=True)
