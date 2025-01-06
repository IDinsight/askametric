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

### 3. Set up the .env file

In the root directory, create a `.env` file and add the following variables:

```
OPENAI_API_KEY=<your_openai_api_key>
```

### 4. Run the demo notebook

Run Jupyter Notebook:

```
jupyter notebook
```

And open the `demo.ipynb` file and run the cells to see the code in action!

### 5. Evaluating the pipeline

Open and run the `validation.ipynb` cells to see how to evaluate the responses from the pipeline for metrics like Relevancy, Consistenty, Accuracy, etc.

You can also use the `validate.py` script to evaluate the pipeline in a faster and more automated way.

To run the validate.py script, go through the following steps:

1. Inside the validation folder, create 2 folders - `test_cases`, and `results`.
2. Add sqlite database test files to whichever folder you want, say the `databases` folder at the root directory.
3. Add the test cases in the `test_cases` folder. Remember to keep filenames of the sqlite databases and the corresponging test_cases files the same. Example, tn_covid.sqlite and tn_covid.csv.
4. In the validation folder, create a .env file with the following variables:

```
<<<

Use the validation_template.env
file for reference

>>>
```

5. Open up the terminal in the root director and run the following command:

```
make validate
```

6. To get summary results, open the `validation/validation_analysis.ipynb` notebook and run the cells. You can also run your custom analysis on the results in this notebook.

_Note: This repository is a work-in-progress. We are continuously improving the code and documentation to help you use and further build on this code easily._
