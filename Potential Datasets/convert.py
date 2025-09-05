import json
import os
import pandas as pd

# Map ContractNLI labels to your schema
label_map = {
    "Entailment": "compliant",
    "Contradiction": "noncompliant",
    "NotMentioned": "uncertain" 
}

def load_and_convert(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Hypothesis mapping (nda-1 -> hypothesis text)
    hypothesis_map = {k: v["hypothesis"] for k, v in data["labels"].items()}

    rows = []
    for doc in data["documents"]:
        text = doc["text"]
        spans = doc["spans"]
        annotations = doc["annotation_sets"][0]["annotations"]

        for hypo_id, ann in annotations.items():
            choice = ann["choice"]
            label = label_map.get(choice)
            # if label is None:
            #     continue  # skip NotMentioned

            reference_clause = hypothesis_map.get(hypo_id, hypo_id)

            # Collect evidence spans
            evidence_texts = []
            for span_idx in ann["spans"]:
                if 0 <= span_idx < len(spans):
                    start, end = spans[span_idx]
                    evidence_texts.append(text[start:end])
            target_clause = " ".join(evidence_texts).strip()

            if target_clause:
                rows.append({
                    "reference_clause": reference_clause,
                    "target_clause": target_clause,
                    "compliance": label,
                    "source_file": doc["file_name"]
                })

    return rows

# Process all splits
all_rows = []
for split in ["train.json", "dev.json", "test.json"]:
    if os.path.exists(split):
        all_rows.extend(load_and_convert(split))

# Save as CSV
df = pd.DataFrame(all_rows)
df.to_csv("contract_compliance_dataset.csv", index=False, encoding="utf-8")

print(f"Converted {len(df)} (hypothesis, evidence) pairs into contract_compliance_dataset.csv")
