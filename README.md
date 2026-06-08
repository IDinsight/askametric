![maintenance-status](https://img.shields.io/badge/maintenance-as--is-yellow.svg)


# Ask a Metric

Ask-a-Metric is a Text-2-SQL python package which helps you answer questions using information from databases.

- 📌 [What is Ask-a-Metric?](https://www.idinsight.org/article/ask-a-metric-your-ai-data-analyst-on-whatsapp/)
- 💬 [Chat with AAM!](https://demo.askametric.com/)
- ⚙️ [How we built Ask-a-Metric](https://idinsight.github.io/tech-blog/blog/aam_pseudo_agent/)

## Getting started

Start with a fresh environment - either conda or venv.

Use Python 3.11.

### 1. Open your preferred IDE and clone this repository:

```
git clone https://github.com/IDinsight/askametric
```

### 2. Install packages using Poetry.

If you have conda, run the following:

```
conda install poetry
```

Else, run:

```
curl -sSL https://install.python-poetry.org | python3 -
```

Confirm Poetry installation by running:

```
poetry --version
```

Assuming we already have an environment set up:

```
poetry config virtualenvs.create false
```

And finally, install all the required packages:

```
poetry install
```

### 3.a. Set up the .env file

In the root directory, create a `.env` file and add the following variables:

For OpenAI models:

```
OPENAI_API_KEY=<your_openai_api_key>
```

For AWS Bedrock models (e.g., Claude):

```
AWS_ACCESS_KEY_ID=<your_aws_access_key_id>
AWS_SECRET_ACCESS_KEY=<your_aws_secret_access_key>
AWS_REGION=<your_aws_region>
```

### 3.b. Manually add LLM configuration

Alternatively, you can add the LLM configuration in the demo notebook directly. This way
is more relevant for running askametric in a server environment.

The notebook has sections where you can choose which provider to test.

### 4. Run the demo notebook

Run Jupyter Notebook:

```
jupyter notebook
```

And open the `demo.ipynb` file and run the cells to see the code in action!

### 5. Switching between models

You can configure the model to use by setting the `llm` parameter in the LLMQueryProcessor. For example:

- To use OpenAI GPT-4o:

  ```python
  llm = "gpt-4o"
  ```

- To use AWS Claude 3.7 via Bedrock (Assuming the region is set to `us`):
  ```python
  llm = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
  ```

Ensure the corresponding environment variables are set in the `.env` file.

Additionally, you can specify different models for various components in the query processor:

- **Primary LLM**: Set the `llm` variable for the main language model.
- **Guardrails**: Set the `guardrails_llm` variable for enforcing constraints or rules.
- **Validation**: Set the `validation_llm` variable for validating responses.

Example:

```python
llm = "gpt-4o"
guardrails_llm = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
validation_llm = "gpt-4o"

qp = LLMQueryProcessor(
    query,
    session,
    metric_db_id,
    db_type,
    llm,
    guardrails_llm,
    sys_message,
    db_description,
    column_description="",
    num_common_values=num_common_values,
    indicator_vars=indicator_vars,
)
```

This allows you to mix and match models based on your requirements.

### 6. Evaluating the pipeline

Open and run the `validation.ipynb` cells to see how to evaluate the responses from the pipeline for metrics like Relevancy, Consistency, Accuracy, etc.

You can also use the `validate.py` script to evaluate the pipeline in a faster and more automated way.

To run the validate.py script, go through the following steps:

1. Inside the validation folder, create 2 folders - `test_cases`, and `results`.
2. Add sqlite database test files to whichever folder you want, say the `databases` folder at the root directory.
3. Add the test cases in the `test_cases` folder. Remember to keep filenames of the sqlite databases and the corresponding test_cases files the same. Example, tn_covid.sqlite and tn_covid.csv.
4. In the validation folder, create a .env file with the following variables:

```
<<<

Use the validation_template.env
file for reference

>>>
```

5. Open up the terminal in the root directory and run the following command:

```
make validate
```

6. To get summary results, open the `validation/validation_analysis.ipynb` notebook and run the cells. You can also run your custom analysis on the results in this notebook.

_Note: This repository is a work-in-progress. We are continuously improving the code and documentation to help you use and further build on this code easily._
