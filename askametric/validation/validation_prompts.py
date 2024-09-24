# Prompts for the validation pipeline


def grading_bot_prompt():
    """
    System prompt for the grading bot
    """
    prompt = """
    You are a grading bot. You will be asked to evaluate messages that
    you receive, based on some criteria.
    You should give a grade of 1 if the message satisfies the given
    criteria, and 0 if it does not.

    You should respond in the following json format:
    {"score": your grade, "reason": reason for your grading}
    """
    return prompt


def get_relevancy_prompt(question: str, llm_response: str):
    """
    Create prompt to check relevancy of generated response.
    """
    prompt = f"""
    Here is the message to evaluate
    ---- Message Begins----------------
    Question: {question}
    Answer: {llm_response}
    ---- Message Ends------------------

    ---- Evaluation Criteria ----
    Does "Answer" address the key elements of the "Question"?
    """
    return prompt


def get_accuracy_prompt(correct_answer: str, llm_response: str):
    """
    Create prompt to check accuracy of generated response.
    """
    prompt = f"""
    Here is the message to evaluate
    ----Message Begins----------------
    Correct Answer: {correct_answer}
    Answer: {llm_response}
    ----Message Ends----------------

    ---- Evaluation Criteria ----
    Is the "Answer" similar in meaning to "Correct Answer"?
    Remember, the "Answer" and "Correct Answer" ONLY NEED TO BE
    SIMILAR in general meaning.
    """
    return prompt


def get_instructions_prompt(question: str, instructions: str, llm_response: str):
    """
    Create prompt to check whether the answer follows instructions
    """
    prompt = f"""
    Here is the message to evaluate
    ---- Message Begins----------------
    Question: {question}
    Instructions: {instructions}
    Answer: {llm_response}
    ---- Message Ends----------------

    ---- Evaluation Criteria -----
    Given the "Question", does the "Answer" follow the "Instructions"?
    """
    return prompt


def get_consistency_prompt(question: str, llm_response: str):
    """
    Create prompt to check consistency of generated response.
    """
    prompt = f"""
    Here is the message to evaluate
    ---- Message Begins----------------
    Question: {question}
    Answer: {llm_response}
    ---- Message Ends----------------

    ---- Evaluation Criteria -----
    Is the format of "Answer" consistent with what the "Question" asks?
    For e.g. if the "Question" asks to list something, "Answer" MUST contain
    bullet points or a number list. If "Question" asks for ranking by an indicator,
    "Answer" MUST contain a ranked list.
    """
    return prompt
