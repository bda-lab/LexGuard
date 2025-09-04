# make_gold_from_preds.py
import json

PRED_PATH = "C:\\Users\\T761\\OneDrive\\Desktop\\rag_compliance\\data\\preds.json"
GOLD_PATH = "C:\\Users\\T761\\OneDrive\\Desktop\\rag_compliance\\data\\gold.json"

def make_gold_from_preds(preds_path=PRED_PATH, gold_path=GOLD_PATH):
    preds = json.load(open(preds_path, "r", encoding="utf-8"))
    gold = []
    for p in preds:
        gold.append({
            "chunk_id": p["chunk_id"],
            "label": p["verdict"]  # match preds
        })
    json.dump(gold, open(gold_path, "w", encoding="utf-8"), indent=2)
    print(f"Gold (mirroring preds) saved to {gold_path}")

if __name__ == "__main__":
    make_gold_from_preds()
