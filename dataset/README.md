# langchain docs dataset

The dataset can be downloaded from <https://dl.kylrth.com/langchain_docs.tar.gz>.

## building the dataset

To build the dataset yourself, first install the requirements with `pip install -r requirements.txt`. Then run `build.sh`. The following sections detail what the each of the Python scripts do when the script calls them.

## [scrape.py](scrape.py)

[scrape.py](scrape.py) scrapes the [LangChain Python docs](https://python.langchain.com). This creates a folder called `docs/` with the following structure:

```txt
docs/
  concepts/
    docs/modules/agents.md
    ...
  procedures/
    full/
      docs/modules/agents/how_to/agent_vectorstore.md
      ...
```

The `is_procedure(url: str) -> bool` function in [scrape.py](scrape.py) determines whether a document ends up in `concepts/` or `procedures/full/`.

## [format_procedure.py](format_procedure.py)

[format_procedure.py](format_procedure.py) runs all the procedures through GPT-4 to remove all unnecessary details and structure the procedures into a set of steps.
The formatted procedures will be stored under `docs/procedures/formatted/`.

The `count_markdown_files(path: str) -> int` function in [format_procedure.py](format_procedure.py) counts the number of markdown files in the specified directory, allowing you to get the total number of procedures in this dataset for any potential use.
The `count_tokens(path: str) -> int` function in [format_procedure.py](format_procedure.py) counts the number of tokens in all the markdown files in the specified directory, allowing you to approximate the cost of running this pipeline using OpenAI's pricing.

## [api_ref.py](api_ref.py)

[api_ref.py](api_ref.py) checks out the LangChain repo, builds the docs, cleans up the formatting, and stores the resulting files in `docs/api`.
