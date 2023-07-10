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
