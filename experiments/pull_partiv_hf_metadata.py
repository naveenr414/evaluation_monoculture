"""Download public Hugging Face metadata for the Part IV open-model sample.

The script is resumable. It stores only public repository metadata; the local token
is read from ``hf_token.txt`` and is never written to the output.
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests


LABS = {
    "qwen", "google", "meta-llama", "mistralai", "deepseek-ai", "microsoft",
    "01-ai", "ibm-granite", "allenai", "tiiuae", "eleutherai",
}
OUT = Path("data/partiv/hf_partiv_model_metadata.jsonl")


def sample_model_ids():
    scores = pd.read_parquet("data/ollm_subtasks.parquet")
    scores["key"] = scores.model.str.lower()
    meta = pd.read_parquet("data/benchmarks.parquet")
    meta["key"] = meta.hf_model_id.str.lower()
    joined = scores[["key"]].merge(meta[["key", "hf_model_id"]], on="key")
    org = joined.hf_model_id.str.split("/").str[0].str.lower()
    return joined.loc[org.isin(LABS), "hf_model_id"].drop_duplicates().tolist()


def completed_ids():
    if not OUT.exists():
        return set()
    with OUT.open() as f:
        return {json.loads(line)["requested_id"] for line in f if line.strip()}


def main():
    token_path = Path("hf_token.txt")
    if not token_path.is_file() or not token_path.read_text().strip():
        raise SystemExit("hf_token.txt is required")
    token = token_path.read_text().strip()
    done = completed_ids()
    targets = [m for m in sample_model_ids() if m not in done]
    print(f"metadata records already saved: {len(done)}; remaining: {len(targets)}")

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as out:
        for n, model_id in enumerate(targets, 1):
            response = session.get(f"https://huggingface.co/api/models/{model_id}", timeout=30)
            record = {"requested_id": model_id, "status_code": response.status_code}
            if response.ok:
                data = response.json()
                record.update({
                    "id": data.get("id"),
                    "sha": data.get("sha"),
                    "last_modified": data.get("lastModified"),
                    "created_at": data.get("createdAt"),
                    "tags": data.get("tags"),
                    "pipeline_tag": data.get("pipeline_tag"),
                    "library_name": data.get("library_name"),
                    "card_data": data.get("cardData"),
                    "config": data.get("config"),
                })
            else:
                record["error"] = response.text[:500]
            out.write(json.dumps(record) + "\n")
            out.flush()
            if n % 25 == 0 or n == len(targets):
                print(f"saved {n}/{len(targets)}")
            time.sleep(0.36)


if __name__ == "__main__":
    main()
