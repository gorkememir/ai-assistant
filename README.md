# Inventory Assistant

A simple **AI-powered inventory assistant** that connects to a Google Sheet, creates embeddings of each item/location/quantity, and lets you query your inventory in natural language.  

Built with:
- **Google Sheets API (gspread)**
- **Pandas**
- **FAISS** for vector search
- **OpenAI embeddings + LLM** (or local LLM via API)
- **Dotenv** for configuration

---

## 🚀 Features
- Load inventory from a Google Sheet (`item`, `storage`, `quantity`, `updatedAt`).
- Generate **vector embeddings** for each row (cached in `embeddings_cache.json`).
- Store embeddings in a **FAISS index** for fast semantic search.
- Ask natural questions like:
  - *"Where are the AA batteries?"*
  - *"How many HDMI cables do I have?"*
- Handles **duplicates** (sums quantities instead of flagging errors).
- Can connect to **OpenAI API** or a **local LLM server**.

---

## ⚙️ Setup

### Clone repo & install dependencies
```bash
git clone <your-repo-url>
cd inventory-assistant
pip install -r requirements.txt
