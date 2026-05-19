from fenn.rag import RAG
from fenn import Fenn

app = Fenn()
open_router_key = app.get_environ("OPEN_ROUTER_KEY")
    
@app.entrypoint
def main(args):

    rag = RAG(model=args["rag"]["model"],
              model_provider=args["rag"]["provider"],
              model_api_key=open_router_key)
    
    rag.add_source("knowledge/docs.md")

    print(rag.chat("What is fenn?"))

if __name__ == "__main__":
    app.run()