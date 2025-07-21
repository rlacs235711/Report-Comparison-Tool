import requests, json, textwrap, tqdm, re
import pandas as pd

URL   = "http://localhost:11434/api/chat"

def extract_Ollama(prompt: str, text: str, model: str) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user",   "content": text[:30000]}
        ]}
    r = requests.post(URL, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]