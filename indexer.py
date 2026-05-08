"""
Buster Knowledge Indexer
Reads all documents from the Buster Knowledge Base Google Drive folder,
chunks them, and upserts into Pinecone for RAG retrieval.

Run this script:
1. When setting up Buster for the first time
2. Whenever documents in the knowledge base are updated

Usage:
    python indexer.py

Environment variables required:
    PINECONE_API_KEY
    PINECONE_INDEX_NAME (default: buster-knowledge)
    GOOGLE_CREDENTIALS_JSON
    KNOWLEDGE_FOLDER_ID
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
KNOWLEDGE_FOLDER_ID = os.environ["KNOWLEDGE_FOLDER_ID"]   # 1yZTK7gw-SLwTTu3CRkFSsuVHamVcR39c

CHUNK_SIZE          = 800    # characters per chunk
CHUNK_OVERLAP       = 150    # overlap between chunks
NAMESPACE           = "buster-docs"
# ─────────────────────────────────────────────────────────────

# Files to skip (non-document files in the folder)
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
        scopes=[
            "https://www.googleapis.com/auth/drive.readonly"
        ]
    )
    return build("drive", "v3", credentials=creds)

def list_folder_files(service, folder_id):
    """List all files in a folder, resolving shortcuts to their targets."""
    results = service.files().list(
        q=f"'{folder_id}' in parents",
        fields="files(id, name, mimeType, shortcutDetails)",
        pageSize=50
    ).execute()

    files = []
    for f in results.get("files", []):
        if f["name"] in SKIP_TITLES:
            continue

        # Resolve shortcuts to actual file
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
    """Download a file and extract its text content."""
    text = ""

    try:
        # Google Docs -> export as plain text
        if mime_type == "application/vnd.google-apps.document":
            response = service.files().export(
                fileId=file_id,
                mimeType="text/plain"
            ).execute()
            text = response.decode("utf-8") if isinstance(response, bytes) else response

        # Google Sheets -> export as CSV
        elif mime_type == "application/vnd.google-apps.spreadsheet":
            response = service.files().export(
                fileId=file_id,
                mimeType="text/csv"
            ).execute()
            text = response.decode("utf-8") if isinstance(response, bytes) else response

        # PDF -> download and extract with PyMuPDF
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

        # Word docs (.docx)
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

        # Plain text
        elif mime_type in ("text/plain", "text/csv"):
            request = service.files().get_media(fileId=file_id)
            buf = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            text = buf.getvalue().decode("utf-8", errors="replace")

        else:
            print(f"  Skipping unsupported mime type: {mime_type} for {file_name}")
            return ""

    except Exception as e:
        print(f"  Error downloading {file_name}: {e}")
        return ""

    return text.strip()

# ── TEXT CHUNKING ─────────────────────────────────────────────
def chunk_text(text, source_name, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks.
    Tries to split on paragraph/sentence boundaries where possible.
    """
    if not text:
        return []

    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks = []
    current_chunk = []
    current_length = 0
    chunk_index = 0

    for para in paragraphs:
        para_length = len(para)

        # If this paragraph alone exceeds chunk size, split it by sentences
        if para_length > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                if current_length + len(sentence) > chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append({
                        "text": " ".join(current_chunk),
                        "source": source_name,
                        "section": f"chunk_{chunk_index}",
                        "chunk_index": chunk_index
                    })
                    chunk_index += 1
                    # Keep overlap
                    overlap_words = " ".join(current_chunk).split()[-30:]
                    current_chunk = [" ".join(overlap_words), sentence]
                    current_length = len(" ".join(current_chunk))
                else:
                    current_chunk.append(sentence)
                    current_length += len(sentence)
        else:
            if current_length + para_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    "text": " ".join(current_chunk),
                    "source": source_name,
                    "section": f"chunk_{chunk_index}",
                    "chunk_index": chunk_index
                })
                chunk_index += 1
                # Keep overlap
                overlap_words = " ".join(current_chunk).split()[-30:]
                current_chunk = [" ".join(overlap_words), para]
                current_length = len(" ".join(current_chunk))
            else:
                current_chunk.append(para)
                current_length += para_length

    # Save final chunk
    if current_chunk:
        chunks.append({
            "text": " ".join(current_chunk),
            "source": source_name,
            "section": f"chunk_{chunk_index}",
            "chunk_index": chunk_index
        })

    return chunks

# ── PINECONE UPSERT ───────────────────────────────────────────
def setup_pinecone_index(pc):
    """Create Pinecone index if it doesn't exist, using integrated inference."""
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing_indexes:
        print(f"Creating new Pinecone index: {INDEX_NAME}")
        pc.create_index_for_model(
            name=INDEX_NAME,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "multilingual-e5-large",
                "field_map": {"text": "text"}
            }
        )
        # Wait for index to be ready
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            print("Waiting for index to be ready...")
            time.sleep(2)
        print(f"Index {INDEX_NAME} ready")
    else:
        print(f"Using existing index: {INDEX_NAME}")

    return pc.Index(INDEX_NAME)

def upsert_chunks(index, chunks, source_name):
    """Upsert chunks into Pinecone with integrated embeddings."""
    if not chunks:
        return

    # Prepare records for Pinecone integrated inference
    records = []
    for chunk in chunks:
        record_id = f"{source_name.replace(' ', '_').replace('/', '_')}_chunk_{chunk['chunk_index']}"
        records.append({
            "_id": record_id,
            "text": chunk["text"],
            "source": chunk["source"],
            "section": chunk["section"],
        })

    # Upsert in batches of 50
    batch_size = 50
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        index.upsert_records(NAMESPACE, batch)
        print(f"  Upserted batch {i//batch_size + 1}/{(len(records)-1)//batch_size + 1} ({len(batch)} chunks)")
        time.sleep(0.5)  # Rate limit protection

# ── MAIN INDEXING FLOW ────────────────────────────────────────
def main():
    print("=" * 60)
    print("BUSTER KNOWLEDGE INDEXER")
    print("=" * 60)

    # Initialize Pinecone
    print("\nConnecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = setup_pinecone_index(pc)

    # Initialize Google Drive
    print("Connecting to Google Drive...")
    service = get_drive_service()

    # List all files in the knowledge base folder
    print(f"\nScanning folder: {KNOWLEDGE_FOLDER_ID}")
    files = list_folder_files(service, KNOWLEDGE_FOLDER_ID)
    print(f"Found {len(files)} documents to index")

    total_chunks = 0

    for file_info in files:
        file_name = file_info["name"]
        file_id = file_info["id"]
        mime_type = file_info["mimeType"]

        print(f"\nProcessing: {file_name}")
        print(f"  Type: {mime_type}")

        # Extract text
        text = download_file_as_text(service, file_id, mime_type, file_name)

        if not text:
            print(f"  No text extracted - skipping")
            continue

        print(f"  Extracted {len(text):,} characters")

        # Chunk the text
        chunks = chunk_text(text, file_name)
        print(f"  Created {len(chunks)} chunks")

        # Upsert to Pinecone
        upsert_chunks(index, chunks, file_name)
        total_chunks += len(chunks)

        # Small delay between documents
        time.sleep(1)

    print("\n" + "=" * 60)
    print(f"INDEXING COMPLETE")
    print(f"Total documents processed: {len(files)}")
    print(f"Total chunks indexed: {total_chunks}")
    print(f"Pinecone index: {INDEX_NAME} / namespace: {NAMESPACE}")
    print("=" * 60)

if __name__ == "__main__":
    main()
