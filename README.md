# analogy-augmented generation (AAG)

This project implements and evaluates the AAG method described in our paper.

Procedures are defined like this:

```python
class Procedure:
    _input: str
    output: str
    steps: list[str]
```

## tasks

### RecipeNLG

(evaluation on a "normal" LM (T5, BERT, small Llama 2, etc.) because this task is too easy for LLMs)

```python
recipe = Procedure(
    _input="flour, milk, eggs, vanilla, sugar",
    output="crèpe",
    steps=[
        "set the skillet to medium heat",
        ...
    ],
)
```

### LCStep

We have created a dataset called LCStep containing LangChain tutorials condensed into concise steps. See [dataset/README.md](dataset/README.md) for details on creating LCStep. We will leverage LLMs to do perform retrieval-augmented generation (RAG), following several approaches:

(evaluation on a coding LLM)

```python
tut = Procedure(
      _input="dataset of Wikipedia articles",
      output="question-answering RAG system",
      steps=[
          "instantiate a model such as langchain.llms.OpenAI",
          ...
      ]
)
```

### math

[JEEBench](https://github.com/dair-iitd/jeebench)

(evaluation on a (math?) LLM)

```python
solution = Procedure(
    _input="word problem",
    output="answer",
    steps=[
        "We know that the train is traveling at 40km/h...",
        ...
    ]
)
```

## baselines

- **zero**: model prompted zero-shot to generate steps
- **RAG**: model augmented with retrieval over the supporting documents. This may benefit from retrieval optimizations like [HyDE](https://arxiv.org/abs/2212.10496), [I^3](https://arxiv.org/abs/2306.02371), or [EAR](https://arxiv.org/abs/2305.17080).
- **AAG**: (our method)
- **teacher forcing (oracle)**: Instead of retrieving over its own generated procedures, the system now retrieves over the gold procedures from the dataset for the items it has already seen.

## evaluation

Model-based evaluation, probably voting among several LLMs that compare the generation with the gold standard.

## dependencies

```sh
pip install -r requirements.txt
```

If you want CPU versions of PyTorch (a dependency of `sentence-transformers`), be sure to install them beforehand. If you're doing development, also run `pip install -r requirements_dev.txt`.

## running experiments

To run an experiment, you need to set up a Weaviate instance to function as the vector store. You can run `docker compose up -d` to start this service in the background. Then to generate results for the RAG method on 10 samples, run the following:

```sh
OPENAI_API_KEY=$(cat openai.key) python run.py --system RAG -n 10
```

Of course, you'll need to have your API key in `openai.key` for this to work. The generated results will be in `output.csv`, but they're available in a more human-readable format under `logs/`.

## development status

- [ ] update definition of a procedure
- [ ] get away from Docker (use embedded)
- datasets
  - [ ] RecipeNLG
  - [ ] LCStep
  - [ ] some other code dataset?
  - [ ] JEEBench(?)
- method
  - [x] implement RAG
  - [ ] implement another baseline (Active RAG)?
  - [ ] implement another baseline specific to the dataset
  - [ ] implement the memory system
  - [ ] implement AAG
  - [ ] add flags for ablations to AAG
- evaluation
  - [x] implement experiment framework (Do we want a train/test split for LCStep or are we good with gradual learn/eval?)
  - [ ] add metrics (including dataset-specific ones)
  - [ ] think about tables/figures we want to include
    - [ ] a figure demonstrating that once the procedural memory is robust the system refers to external documents less
