import streamlit as st
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import pypdf
import uuid
from groq import Groq

# --- CONFIG ---
QDRANT_HOST = "http://localhost:6333" # Update with your Qdrant host if needed
QDRANT_COLLECTION = "medical_docs"
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500  # characters
CHUNK_OVERLAP = 100

# --- INIT CLIENTS ---
qdrant = QdrantClient(url=QDRANT_HOST)
embedder = SentenceTransformer(EMBEDDING_MODEL)

def ensure_collection():
    if not qdrant.collection_exists(QDRANT_COLLECTION):
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=384, distance="Cosine")
        )

# --- HELPERS ---
def pdf_to_text(pdf_file):
    reader = pypdf.PdfReader(pdf_file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def hash_doc_chunk(source_name, chunk):
    # Qdrant needs point IDs as unsigned integer or a UUID string
    return str(uuid.uuid4())

def upsert_chunks_to_qdrant(source_name, chunks):
    ensure_collection()
    embeddings = embedder.encode(chunks).tolist()
    payloads = [
        {"source": source_name, "chunk": c, "chunk_id": i}  # Removed hash, not needed
        for i, c in enumerate(chunks)
    ]
    # Generate valid UUIDs for the ids
    ids = [hash_doc_chunk(source_name, c) for c in chunks]
    qdrant.upsert(
        collection_name=QDRANT_COLLECTION,
        points=models.Batch(ids=ids, vectors=embeddings, payloads=payloads)
    )

# --- PAGE LAYOUT ---
st.title("Medical RAG App")

# --- PDF UPLOAD + INGESTION ---
pdf_files = st.file_uploader(
    "Upload medical PDF(s)", type="pdf", accept_multiple_files=True
)

if pdf_files:
    for pdf_file in pdf_files:
        st.markdown(f"#### {pdf_file.name}")
        try:
            text = pdf_to_text(pdf_file)
            chunks = chunk_text(text)
            upsert_chunks_to_qdrant(pdf_file.name, chunks)
            st.success(f"Ingested {len(chunks)} chunks from {pdf_file.name}")
        except Exception as e:
            st.error(f"Error processing {pdf_file.name}: {e}")

# --- QUESTION INPUT ---
question = st.text_input("Ask a medical question:")

def search_top_k_chunks(query, k=5):
    ensure_collection()
    q_emb = embedder.encode([query]).tolist()[0]
    qdrant_results = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=q_emb,
        limit=k
    )
    # Each result is a ScoredPoint
    top_chunks = []
    for pt in qdrant_results:
        payload = pt.payload
        chunk_text = payload.get("chunk", "")
        source = payload.get("source", "Unknown")
        chunk_id = payload.get("chunk_id", None)
        top_chunks.append({
            "chunk": chunk_text,
            "source": source,
            "chunk_id": chunk_id
        })
    return top_chunks

def build_prompt(query, chunks):
    context = "\n\n".join([
        f"[{c['source']} - chunk {c['chunk_id']}]:\n{c['chunk']}" for c in chunks
    ])
    prompt = (
        f"Use the following reference text chunks to answer the user question.\n\n "
        f"References: {context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    return prompt

def ask_llama_groq(prompt):
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant. Always cite sources using the source and chunk id. Answer concisely."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()

answer = None
refs = []

if st.button("Submit") and question:
    with st.spinner("Getting answer from Groq LLaMA 70B..."):
        try:
            top_chunks = search_top_k_chunks(question, k=5)
            if not top_chunks:
                st.error("No relevant medical documents found.")
            else:
                prompt = build_prompt(question, top_chunks)
                answer = ask_llama_groq(prompt)
                unique_sources = list({f"{c['source']} (chunk {c['chunk_id']})" for c in top_chunks})  # set for unique, list for order
                refs = unique_sources
        except Exception as e:
            st.error(f"Error during retrieval or LLM call: {e}")

# --- ANSWER & REFERENCES REGION ---
st.markdown("---")
st.subheader("Answer:")
if answer:
    st.write(answer)

st.subheader("References:")
if refs:
    for ref in refs:
        st.write(f"- {ref}")
