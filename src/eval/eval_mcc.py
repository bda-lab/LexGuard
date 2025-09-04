# eval_mcc.py
import json
import argparse
import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix, accuracy_score

PRED_PATH = "C:\\Users\\T761\\OneDrive\\Desktop\\rag_compliance\\data\\preds.json"
GOLD_PATH = "C:\\Users\\T761\\OneDrive\\Desktop\\rag_compliance\\data\\gold.json"
def to_bin(label):
    if isinstance(label, str):
        return 1 if label.lower() == "fail" else 0
    return int(label)

def load_labels(pred_path, gold_path):
    preds = json.load(open(pred_path, "r", encoding="utf-8"))
    golds = json.load(open(gold_path, "r", encoding="utf-8"))

    # gold can be list[dict] with "chunk_id"/"label" or list[int]/list[str]
    if isinstance(golds, list) and len(golds) > 0 and isinstance(golds[0], dict):
        gold_map = {g["chunk_id"]: to_bin(g["label"]) for g in golds}
    else:
        gold_map = {i: to_bin(v) for i, v in enumerate(golds)}

    y_true, y_pred = [], []
    for p in preds:
        cid = p["chunk_id"]
        if cid in gold_map:
            y_true.append(gold_map[cid])
            y_pred.append(to_bin(p["verdict"]))
    return np.array(y_true), np.array(y_pred)

def safe_mcc(y_true, y_pred, policy="sklearn"):
    """
    Handle single-class edge cases:
      - 'sklearn' : return sklearn's value (0.0) and warning
      - 'nan'     : return NaN if single-class
      - 'perfect-if-match' : if arrays identical -> 1.0 else 0.0
      - 'formula-zero-div-is-0' : compute MCC formula; if denom==0 -> 0.0
    """
    uniq_true, uniq_pred = np.unique(y_true), np.unique(y_pred)
    single_class = (len(uniq_true) == 1) or (len(uniq_pred) == 1)
    if not single_class:
        return matthews_corrcoef(y_true, y_pred)

    if policy == "sklearn":
        return matthews_corrcoef(y_true, y_pred)  # will warn and return 0.0
    elif policy == "nan":
        return float("nan")
    elif policy == "perfect-if-match":
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0
    elif policy == "formula-zero-div-is-0":
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        return ((tp * tn - fp * fn) / denom) if denom else 0.0
    else:
        raise ValueError(f"Unknown policy: {policy}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=PRED_PATH)
    ap.add_argument("--gold", default=GOLD_PATH)
    ap.add_argument("--policy", default="sklearn",
                    choices=["sklearn", "nan", "perfect-if-match", "formula-zero-div-is-0"],
                    help="How to handle single-class MCC edge cases")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    y_true, y_pred = load_labels(args.preds, args.gold)
    if args.verbose:
        print("y_true:", y_true.tolist())
        print("y_pred:", y_pred.tolist())

    # Basic stats
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print("Counts -> TN:", tn, "FP:", fp, "FN:", fn, "TP:", tp)
    print("Accuracy:", acc)

    # Robust MCC
    mcc = safe_mcc(y_true, y_pred, policy=args.policy)
    print("MCC (policy=%s):" % args.policy, mcc)

if __name__ == "__main__":
    main()
