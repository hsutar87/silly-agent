import requests
from typing import List

def get_embedding(
    text: str,
    model: str,
    url: str,
    timeout: int = 30,
) -> List[float]:
    resp = requests.post(
        url,
        json={"model": model, "input": text},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("embeddings", [[]])[0]
