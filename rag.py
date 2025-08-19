import os
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
import faiss, numpy as np
from openai import OpenAI
import json

load_dotenv()

SCOPE = ["https://www.googleapis.com/auth/spreadsheets"]
SHEET_ID = os.getenv("SHEET_ID", "15geqTummkYJARcVRQBEN9bClgUbPiWhfFRyH8ULs0Yk")
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "Inventory")
GOOGLE_APPLICATION_CREDENTIALS = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

print("[1] Loading Google Sheet...")

def load_sheet_as_df(sheet_id: str, worksheet_name: str) -> pd.DataFrame:
    try:
        creds = Credentials.from_service_account_file(
            GOOGLE_APPLICATION_CREDENTIALS,
            scopes=SCOPE
        )
    except FileNotFoundError:
        raise SystemExit("Service account JSON not found. Check GOOGLE_APPLICATION_CREDENTIALS path.")
    except Exception as e:
        raise SystemExit(f"Failed to load Google credentials: {e}")

    try:
        gc = gspread.authorize(creds)
        ws = gc.open_by_key(sheet_id).worksheet(worksheet_name)
        records = ws.get_all_records()
        return pd.DataFrame(records)
    except gspread.exceptions.WorksheetNotFound:
        raise SystemExit(f"Worksheet '{worksheet_name}' not found in sheet {sheet_id}.")
    except gspread.exceptions.APIError as e:
        raise SystemExit(f"Google Sheets API error: {e}")
    except Exception as e:
        raise SystemExit(f"Unexpected error reading sheet: {e}")


df = load_sheet_as_df(SHEET_ID, WORKSHEET_NAME)
print("[1] Loaded sheet rows:", len(df))

# --- Embedding + FAISS ---
print("[2] Creating embeddings and FAISS index...")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-small")

# docs: one row per item/storage (normalized)
docs = []
for r in df.itertuples():
    item = str(r.item).strip().lower()
    storage = str(r.storage).strip().lower()
    qty = int(r.quantity)
    updated = r.updatedAt
    docs.append({
        "id": f"{item}|{storage}",
        "text": f"Item: {item}. Location: {storage}. Quantity: {qty}. Updated: {updated}."
    })

# --- Embedding cache (persisted) ---
CACHE_FILE = "embeddings_cache.json"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        embedding_cache = json.load(f)
else:
    embedding_cache = {}


def get_embedding(text):
    if text in embedding_cache:
        return embedding_cache[text]
    emb = client.embeddings.create(model=EMB_MODEL, input=[text]).data[0].embedding
    embedding_cache[text] = emb
    with open(CACHE_FILE, "w") as f:
        json.dump(embedding_cache, f)
    return emb


print("[2] Embedding", len(docs), "rows...")
vecs = np.array([get_embedding(d["text"]) for d in docs]).astype("float32")

index = faiss.IndexFlatIP(vecs.shape[1])
faiss.normalize_L2(vecs)
index.add(vecs)

# --- QA Function ---

def answer(question: str):
    print("[3] Embedding question:", question)
    q = get_embedding(question)
    q = np.array([q]).astype("float32"); faiss.normalize_L2(q)

    D, I = index.search(q, 10)
    relevant_threshold = 0.1
    relevant_indices = [i for i, score in zip(I[0], D[0]) if score > relevant_threshold]

    if not relevant_indices:
        return "Sorry, I couldn't find anything relevant to your search. Try rephrasing your question."

    ctx = "\n".join(docs[i]["text"] for i in relevant_indices[:3])

    llm_client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    )

    completion = llm_client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "llama3.1"),
        messages=[
            {"role": "system", "content": """You are a helpful inventory assistant. 
            - Only answer based on the provided context. 
            - If the context doesn't contain relevant information, say so.
            - Always clearly state the item name and its exact location with quantity in a sentence.
            - Keep responses concise and accurate, but provide a natural response. 
            - Feel free to add some jokes occasionally. 
            - When there are duplicate entries;
                - Return the total amount only, don't say it's a duplicate.
                - Don't think there's an error, duplicates are ok."""},
            {"role": "user", "content": f"Question: {question}\n\nContext:\n{ctx}"}
        ]
    )
    ans = completion.choices[0].message.content.strip()
    return ans


if __name__ == "__main__":
    print("[4] Ready. Ask a question...")
    print(answer(input("What u looking for? ")))
