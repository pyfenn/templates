# Fenn

Fenn is a Python framework for managing machine learning and deep learning workflows. It provides reusable project templates, YAML-based configuration management, training abstractions, experiment logging, checkpointing, notifications, and utility tools for data processing.

The library is designed to reduce boilerplate in PyTorch-based projects by standardizing common tasks such as:

- Model training and evaluation
- Experiment configuration
- Logging and tracking
- Checkpoint management
- Notifications (Discord, Slack, Telegram, Email)
- Dataset utilities and profiling

Fenn focuses on improving developer productivity and reproducibility for ML experiments while remaining lightweight and modular.

## Agents Module

The `fenn.agents` module provides a node-based flow framework for building AI agent pipelines. It includes:

- `Flow` — orchestrates nodes connected via `flow.connect(src, dst, action="default")`
- `Node` — base processing unit with `prep`, `exec`, `post` lifecycle hooks
- `LLMClient` — unified provider-agnostic LLM client supporting OpenAI, Anthropic, Gemini, Groq, Ollama, and more
- `RAGNode` — a reusable node that retrieves relevant context from indexed documents using BM25 or FAISS

## LLMClient

`LLMClient` supports any OpenAI-compatible provider. Configure it with:

```python
from fenn.agents import LLMClient

client = LLMClient(provider="openrouter", model="openai/gpt-4o-mini", api_key_env="OPEN_ROUTER_KEY")
answer = client.ask("What is Fenn?")
```

## RAGNode

`RAGNode` loads and indexes documents at construction time, then retrieves the top-k relevant chunks per run:

```python
from fenn.agents import RAGNode, Flow

rag = RAGNode(sources=["knowledge/docs.md"], top_k=5)
flow = Flow(start=rag)
```

## Flow Wiring

Use `flow.connect(src, dst, action)` to wire nodes. Pass `dst=None` to mark an explicit terminal transition (no warning):

```python
flow.connect(node_a, node_b)           # node_a → node_b on "default"
flow.connect(node_b, node_c, "yes")    # node_b → node_c on "yes"
flow.connect(node_b, None, "no")       # node_b → terminal on "no"
```
