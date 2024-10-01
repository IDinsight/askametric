# Ask a Metric

Ask-a-Metric is a Text-2-SQL python package which helps you answer questions using information from databases.

# Demo Usage

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

_Note: This repository is a work-in-progress. We are continuously improving the code and documentation to help you use and further build on this code easily._
