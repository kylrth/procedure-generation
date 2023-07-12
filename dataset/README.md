# langchain docs dataset

The dataset can be downloaded from <https://dl.kylrth.com/langchain_docs.tar.gz>.

## building the dataset

To build the dataset yourself, first install the requirements with `pip install -r requirements.txt`. Then scrape the [LangChain Python docs](https://python.langchain.com) by running `scrape.py`. This creates a folder called `docs/` with the following structure:

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

Next, format the procedures by running `format_procedure.py`. This runs all the procedures through GPT-4 to remove all unnecessary details and structure the procedures into a set of steps.
The formatted procedures will be stored under `docs/procedures/formatted/`.

The `count_markdown_files(path: str) -> int` function in [format_procedure.py](format_procedure.py) counts the number of markdown files in the specified directory, allowing you to get the total number of procedures in this dataset for any potential use.
The `count_tokens(path: str) -> int` function in [format_procedure.py](format_procedure.py) counts the number of tokens in all the markdown files in the specified directory, allowing you to approximate the cost of running this pipeline using OpenAI's pricing.

Finally, run `api_ref.py` to check out the LangChain repo, build the docs, and clean up the formatting. These end up in `docs/api`.
