"""
Buster Knowledge Indexer v2
Reads all documents from the Buster Knowledge Base Google Drive folder,
chunks them, and upserts into Pinecone for RAG retrieval.

Run this script via Render Shell:
    python indexer.py
"""

import os
import json
import time
import io
import re
from pinecone import Pinecone, ServerlessSpec
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import fitz  # PyMuPDF for PDFs
from docx import Document as DocxDocument

# ── CONFIG ────────────────────────────────────────────────────
PINECONE_API_KEY    = os.environ["PINECONE_API_KEY"]
INDEX_NAME          = os.environ.get("PINECONE_INDEX_NAME", "buster-knowledge")
GOOGLE_CREDS_JSON   = os.environ["GOOGLE_CREDENTIALS_JSON"]
KNOWLEDGE_FOLDER_ID = os.environ["KNOWLEDGE_FOLDER_ID"]

CHUNK_SIZE          = 800
CHUNK_OVERLAP       = 150
NAMESPACE           = "buster-docs"
EMBEDDING_MODEL     = "multilingual-e5-large"
EMBEDDING_DIMENSION = 1024  # dimension for multilingual-e5-large
# ─────────────────────────────────────────────────────────────

SKIP_TITLES = {
    "Buster Conversation Memory",
    "Buster Master Knowledge Base",
    "Buster Master Knowledge Base v1",
}

# ── GOOGLE DRIVE SETUP ────────────────────────────────────────
def get_drive_service():
    creds_dict = json.loads(GOOGLE_CREDS_JSON)
    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds)

def list_folder_files(service, folder_id):
    results = service.files().list(
        q=f"'{folder_id}' in parents",
        fields="files(id, name, mimeType, shortcutDetails)",
        pageSize=50
    ).execute()

    files = []
    for f in results.get("files", []):
        if f["name"] in SKIP_TITLES:
            continue
        if f["mimeType"] == "application/vnd.google-apps.shortcut":
            target_id = f.get("shortcutDetails", {}).get("targetId")
            target_mime = f.get("shortcutDetails", {}).get("targetMimeType")
            if target_id:
                files.append({
                    "id": target_id,
                    "name": f["name"],
                    "mimeType": target_mime or f["mimeType"]
                })
        else:
            files.append({
                "id": f["id"],
                "name": f["name"],
                "mimeType": f["mimeType"]
            })
    return files

def download_file_as_text(service, file_id, mime_type, file_name):
    text = ""
    try:
        if mime_type == "application/vnd.google-apps.document":
            response = service.files().export(
                fileId=file_id, mimeType="text/plain"
            ).execute()
            text = response.decode("utf-8") if isinstance(response, bytes) else response

        elif mime_type == "application/vnd.google-apps.spreadsheet":
            response = service.files().export(
                fileId=file_id, mimeType="text/csv"
            ).execute()
            text = response.decode("utf-8") if isinstance(response, bytes) else response

        elif mime_type == "application/pdf":
            request = service.files().get_media(fileId=file_id)
            buf = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            buf.seek(0)
            doc = fitz.open(stream=buf.read(), filetype="pdf")
            pages = []
            for page in doc:
                pages.append(page.get_text())
            text = "\n".join(pages)
            doc.close()

        elif mime_type in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword"
        ):
            request = service.files().get_media(fileId=file_id)
            buf = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            buf.seek(0)
            doc = DocxDocument(buf)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n".join(paragraphs)

        elif mime_type in ("text/plain", "text/csv"):
            request = service.files().get_media(fileId=file_id)
            buf = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            text = buf.getvalue().decode("utf-8", errors="replace")

        else:
            print(f"  Skipping unsupported type: {mime_type}")
            return ""

    except Exception as e:
        print(f"  Error downloading {file_name}: {e}")
        return ""

    return text.strip()

# ── TEXT CHUNKING ─────────────────────────────────────────────
def chunk_text(text, source_name):
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_index = 0

    for para in paragraphs:
        para_length = len(para)

        if para_length > CHUNK_SIZE:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                if current_length + len(sentence) > CHUNK_SIZE and current_chunk:
                    chunks.append({
                        "text": " ".join(current_chunk),
                        "source": source_name,
                        "chunk_index": chunk_index
                    })
                    chunk_index += 1
                    overlap_words = " ".join(current_chunk).split()[-30:]
                    current_chunk = [" ".join(overlap_words), sentence]
                    current_length = len(" ".join(current_chunk))
                else:
                    current_chunk.append(sentence)
                    current_length += len(sentence)
        else:
            if current_length + para_length > CHUNK_SIZE and current_chunk:
                chunks.append({
                    "text": " ".join(current_chunk),
                    "source": source_name,
                    "chunk_index": chunk_index
                })
                chunk_index += 1
                overlap_words = " ".join(current_chunk).split()[-30:]
                current_chunk = [" ".join(overlap_words), para]
                current_length = len(" ".join(current_chunk))
            else:
                current_chunk.append(para)
                current_length += para_length

    if current_chunk:
        chunks.append({
            "text": " ".join(current_chunk),
            "source": source_name,
            "chunk_index": chunk_index
        })

    return chunks

# ── PINECONE SETUP ────────────────────────────────────────────
def setup_pinecone_index(pc):
    existing = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing:
        print(f"Creating new Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Waiting for index to be ready...")
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(2)
        print(f"Index {INDEX_NAME} is ready")
    else:
        print(f"Using existing index: {INDEX_NAME}")

    return pc.Index(INDEX_NAME)

def get_embeddings(texts):
    """Get embeddings using Pinecone's inference API."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    result = pc.inference.embed(
        model=EMBEDDING_MODEL,
        inputs=texts,
        parameters={"input_type": "passage", "truncate": "END"}
    )
    return [item["values"] for item in result]

def upsert_chunks(index, chunks, source_name):
    if not chunks:
        return

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]

        print(f"  Embedding batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}...")
        embeddings = get_embeddings(texts)

        vectors = []
        for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
            record_id = f"{source_name.replace(' ', '_').replace('/', '_')[:50]}_chunk_{chunk['chunk_index']}"
            vectors.append({
                "id": record_id,
                "values": embedding,
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"]
                }
            })

        index.upsert(vectors=vectors, namespace=NAMESPACE)
        print(f"  Upserted {len(vectors)} vectors")
        time.sleep(0.5)

# ── MAIN ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("BUSTER KNOWLEDGE INDEXER v2")
    print("=" * 60)

    print("\nConnecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = setup_pinecone_index(pc)

    print("Connecting to Google Drive...")
    service = get_drive_service()

    print(f"\nScanning folder: {KNOWLEDGE_FOLDER_ID}")
    files = list_folder_files(service, KNOWLEDGE_FOLDER_ID)
    print(f"Found {len(files)} documents to index")

    total_chunks = 0

    for file_info in files:
        file_name = file_info["name"]
        file_id = file_info["id"]
        mime_type = file_info["mimeType"]

        print(f"\nProcessing: {file_name}")

        text = download_file_as_text(service, file_id, mime_type, file_name)
        if not text:
            print(f"  No text extracted - skipping")
            continue

        print(f"  Extracted {len(text):,} characters")

        chunks = chunk_text(text, file_name)
        print(f"  Created {len(chunks)} chunks")

        upsert_chunks(index, chunks, file_name)
        total_chunks += len(chunks)
        time.sleep(1)

    print("\n" + "=" * 60)
    print(f"INDEXING COMPLETE")
    print(f"Documents processed: {len(files)}")
    print(f"Total chunks indexed: {total_chunks}")
    print(f"Index: {INDEX_NAME} / namespace: {NAMESPACE}")
    print("=" * 60)

if __name__ == "__main__":
    main()
