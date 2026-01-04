import numpy as np
import torch

from fenn import Fenn
from fenn.nn.transformers.sequence_classifier import SequenceClassifier

app = Fenn()

@app.entrypoint
def main(args):

    device = args["train"]["device"] if torch.cuda.is_available() else "cpu"

    # Tiny toy dataset (replace with your real samples)
    X_train = [
        "int a=0; if (x) a=1;",                 # pretend: non-vuln
        "char buf[8]; strcpy(buf, input);",     # pretend: vuln
        "if (p) free(p); p=NULL;",              # pretend: non-vuln
        "gets(buf);",                           # pretend: vuln
    ]

    y_train = [0, 1, 0, 1]

    X_test = [
        "memcpy(dst, src, n);",
        "sprintf(buf, user);",
    ]

    y_test = [0, 1]

    cfg = LoRAConfig(
        model_dir=args["llm"]["dir"],    # folder containing your local HF model
        model_name=args["llm"]["name"],  # model name
        device=device,
        epochs=args["train"]["epochs"],
        learning_rate=float(args["train"]["lr"]),
        train_batch_size=2,
        eval_batch_size=2,
        max_length=256,
    )

    clf = SequenceClassifier(cfg)

    # Fit (like sklearn)
    clf.fit(X_train, y_train)

    # Predict labels
    y_pred = clf.predict(X_test)
    print("pred:", y_pred.tolist())

    # Predict probabilities (P(class=0), P(class=1))
    proba = clf.predict_proba(X_test)
    print("proba:", np.round(proba, 4).tolist())

    # Simple accuracy
    acc = clf.score(X_test, y_test)
    print("acc:", acc)

    # Save adapter + tokenizer + metadata
    out_dir = "./artifacts/lora_adapter_run1"
    clf.save(out_dir)

    # Load later (needs base model location too)
    clf2 = SequenceClassifier.load(
        adapter_dir=out_dir,
        base_model_dir="/models",
        base_model_name="my-seqcls-backbone",
        device="cuda"
    )

    print("pred(reloaded):", clf2.predict(X_test).tolist())

if __name__ == "__main__":
    app.run()