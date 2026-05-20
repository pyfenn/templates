"""
RAG chatbot example using the fenn.agents flow system.

Flow:
  RAGNode  →  ChatNode  →  (terminal)

RAGNode retrieves relevant document chunks into shared["rag_context"].
ChatNode calls LLMClient with the full conversation history and context.
The outer while-loop drives multi-turn interaction.
"""

from fenn import Fenn
from fenn.agents import Flow, Node, RAGNode
from fenn.agents.llm import LLMClient

app = Fenn()

_llm_cfg = app.parameters["llm"]
_rag_cfg = app.parameters.get("rag", {})

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer the user's question using ONLY the information provided in the context. "
    "If the answer is not in the context, say you don't know. "
    "Be concise and respond in the same language as the user."
)

class ChatNode(Node):
    """Generates an answer using conversation history + RAG context."""

    def __init__(self, llm: LLMClient):
        super().__init__()
        self._llm = llm
        self._history = []  # list of {"role": ..., "content": ...}

    def prep(self, shared):
        return shared.get("query", ""), shared.get("rag_context", "")

    def exec(self, prep_res):
        query, context = prep_res
        system_msg = SYSTEM_PROMPT
        if context:
            system_msg += f"\n\nContext:\n{context}"

        messages = [{"role": "system", "content": system_msg}]
        messages.extend(self._history)
        messages.append({"role": "user", "content": query})

        return self._llm.chat_complete(messages)

    def post(self, shared, prep_res, answer):
        query, _ = prep_res
        self._history.append({"role": "user",      "content": query})
        self._history.append({"role": "assistant",  "content": answer})
        shared["answer"] = answer
        return "default"

    def reset(self):
        self._history.clear()


def build_flow(llm, rag_cfg):
    rag_node  = RAGNode(
        sources=rag_cfg.get("knowledge_dir", "knowledge"),
        top_k=rag_cfg.get("top_k", 5),
        chunk_mode=rag_cfg.get("chunk_mode", "smart"),
    )
    chat_node = ChatNode(llm)

    flow = Flow(start=rag_node)
    flow.connect(rag_node,  chat_node)
    flow.connect(chat_node, None)       # explicit terminal transition

    return flow, chat_node


def main():
    llm  = LLMClient(
        provider=_llm_cfg.get("provider"),
        model=_llm_cfg.get("model"),
        api_key=app.get_environ("OPEN_ROUTER_KEY"),
    )
    flow, chat_node = build_flow(llm, _rag_cfg)

    print("Chatbot ready. Type 'exit' or press Ctrl+C to quit.")
    print("Type 'reset' to clear conversation history.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if query.lower() == "reset":
            chat_node.reset()
            print("[history cleared]\n")
            continue

        shared = {"query": query}
        flow.run(shared)
        print(f"\nAssistant: {shared['answer']}\n")


if __name__ == "__main__":
    main()
