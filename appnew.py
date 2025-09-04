import streamlit as st
import time
import os
import fitz  # PyMuPDF
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv   # ✅ NEW

# ======================
# Load Environment Variables
# ======================
load_dotenv()  # ✅ This will load variables from your .env file
HF_TOKEN = os.getenv("HF_TOKEN")

# ======================
# HuggingFace Login
# ======================
if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
    except Exception as e:
        st.error(f"Failed to log in to Hugging Face: {e}. Please check your token.")
else:
    st.warning("Hugging Face token not found. Please set the 'HF_TOKEN' environment variable.")

# ======================
# PDF Utility
# ======================
def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ======================
# Embedding + FAISS
# ======================
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    """Splits text into sentences and generates embeddings."""
    sentences = text.split('.')
    embeddings = embedding_model.encode(sentences)
    return sentences, embeddings

def setup_vector_store(embeddings):
    """Creates a FAISS index from text embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_similar_texts(query, index, sentences, top_k=5):
    """Retrieves the most similar sentences from the vector store based on a query."""
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [sentences[i] for i in indices[0]]

# ======================
# HuggingFace IBM Granite
# ======================
@st.cache_resource
def get_model_and_tokenizer():
    """Initializes and caches the model and tokenizer for the IBM Granite model."""
    tokenizer = AutoTokenizer.from_pretrained(
        "ibm-granite/granite-3.3-2b-instruct", token=HF_TOKEN
    )
    model = AutoModelForCausalLM.from_pretrained(
        "ibm-granite/granite-3.3-2b-instruct", token=HF_TOKEN
    )
    return tokenizer, model

def generate_answer(prompt, context):
    """Generates a response using the IBM Granite model based on a prompt and context."""
    tokenizer, model = get_model_and_tokenizer()
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Use the following context to answer the question:\n{context}"},
        {"role": "user", "content": prompt}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True)
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return generated_text.strip()

# ======================
# Streaming Utility
# ======================
def stream_data(text, delay=0.02):
    """Yields words one by one to simulate streaming text."""
    for word in text.split():
        yield word + " "
        time.sleep(delay)

# ======================
# Streamlit App
# ======================
st.title("📘 PDF Q&A with HuggingFace + Granite + FAISS")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    sentences, embeddings = embed_text(pdf_text)
    vector_index = setup_vector_store(embeddings)
    st.success("✅ PDF processed and vector store set up.")

# Input prompt
prompt = st.text_input("Ask a question...")

if prompt and uploaded_file and HF_TOKEN:
    with st.spinner("Thinking..."):
        try:
            relevant_texts = retrieve_similar_texts(prompt, vector_index, sentences)
            context = " ".join(relevant_texts)
            response = generate_answer(prompt, context)
            st.markdown("### Answer")
            st.write_stream(stream_data(response))
        except Exception as e:
            st.error(f"An error occurred: {e}")
