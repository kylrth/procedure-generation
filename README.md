# procedure generation

Our task is generating steps for accomplishing various tasks with the LangChain library. We have created a dataset of LangChain tutorials condensed into concise steps, with each step referencing the LangChain API. See [dataset/README.md](dataset/README.md) for details on the dataset. We will leverage LLMs to do perform retrieval-augmented generation (RAG), following several approaches:

- **zero**: GPT-4 prompted zero-shot to generate steps (should fail due to 2021 knowledge cutoff)
- **RAG**: GPT-4 augmented with retrieval over the API reference documentation plus some LangChain conceptual documentation. This will probably need some retrieval optimizations like [HyDE](https://arxiv.org/abs/2212.10496), [I^3](https://arxiv.org/abs/2306.02371), or [EAR](https://arxiv.org/abs/2305.17080).
- **skill library + environment feedback**: Retrieval now gets access to all procedures generated so far. There will also be environment feedback in the form of a similarity score threshold to recognize when we don't actually have a relevant skill in the library.
- **teacher forcing (oracle)**: Instead of retrieving over its own generated procedures, the system now retrieves over the gold procedures from the dataset for the items it has already seen.

## evaluation

Model-based evaluation, probably voting among several LLMs that compare the generation with the gold standard.

## dependencies

```sh
pip install -r requirements.txt
```

If you want CPU versions of PyTorch (a dependency of `sentence-transformers`), be sure to install them beforehand. If you're doing development, also run `pip install -r requirements_dev.txt`.
