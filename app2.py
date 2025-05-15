import streamlit as st
import pandas as pd
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import requests
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Load and preprocess data
df = pd.read_csv(r"D:\flipkart laptop search engine\data\clean_data.csv")

def create_text(row):
    return (
        f"Product: {row['Product Name']}, Rating: {row['Rating']}, Price: ₹{row['Price']}, "
        f"Processor: {row['Processor']}, RAM: {row['RAM']}, Storage: {row['Storage']}, "
        f"Display: {row['Display']}, OS: {row['OS']}, Warranty: {row['Warranty']}"
    )

df["text"] = df.apply(create_text, axis=1)

# Embed products
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embedding_matrix = model.encode(df["text"].tolist(), show_progress_bar=True).astype("float32")

# FAISS index
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

# --- Search Function ---
def extract_price_limit(query):
    match = re.search(r"under\s*₹?\s*(\d+)", query)
    if match:
        return int(match.group(1))
    return None

def search_products(query, top_k=5):
    price_limit = extract_price_limit(query)
    
    filtered_df = df
    if price_limit:
        filtered_df = df[df['Price'].astype(str).str.replace(",", "").astype(float) <= price_limit]

    if filtered_df.empty:
        return []

    query_vector = model.encode([query]).astype("float32")
    filtered_embeddings = model.encode(filtered_df["text"].tolist()).astype("float32")

    temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
    temp_index.add(filtered_embeddings)

    distances, indices = temp_index.search(query_vector, min(top_k, len(filtered_df)))
    filtered_data = filtered_df.to_dict(orient="records")
    return [filtered_data[i] for i in indices[0]]

# --- LLM Response ---
def generate_response(query, retrieved_products, model_name="llama3-8b-8192", groq_api_key=api_key):
    if not retrieved_products:
        return "❌ Sorry, no matching laptops found for your query."

    product_lines = []
    for i, p in enumerate(retrieved_products, 1):
        summary = (
            f"A budget-friendly laptop with a {p['Display']} display, "
            f"{p['Processor']} processor, and {p['RAM']} RAM."
        )
        product_lines.append(
            f"{i}\nProduct Name: {p['Product Name']}\n"
            f"Price: ₹{p['Price']}\n"
            f"Rating: {p['Rating']}\n"
            f"Summary: {summary}\n"
        )

    context = "\n".join(product_lines)

    prompt = f"""
🔍 Top Products:

Here are the top {len(retrieved_products)} relevant products that match the user's query:

{context}

Based only on these results, recommend the most suitable laptop. Do not use any external knowledge.
"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful shopping assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ API Error {response.status_code}: {response.text}"
    
# --- Streamlit UI ---
st.set_page_config(page_title="Laptop Recommender", layout="wide")
st.title("💻 AI-Powered Laptop Recommendation Assistant")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Enter your laptop requirements 👇", placeholder="e.g. best laptop under 50000 for gaming")

if st.button("🔍 Find Laptops"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching and generating recommendations..."):
            top_k_results = search_products(query)

            if not top_k_results:
                st.error("❌ No laptops found matching your query.")
            else:
                # Save user query in history
                st.session_state.chat_history.append({"role": "user", "content": query})

                st.subheader("🔍 Top Matching Laptops")
                for i, p in enumerate(top_k_results, 1):
                    st.markdown(f"{i}. {p['Product Name']}")
                    st.markdown(f"💰 Price: ₹{p['Price']}")  
                    st.markdown(f"⭐ Rating: {p['Rating']}")
                    st.markdown(f"🧠 Processor: {p['Processor']}, 💾 RAM: {p['RAM']}, 💽 Storage: {p['Storage']}")
                    st.markdown("---")

                # Chatbot recommendation
                st.subheader("🤖 Chatbot Recommendation")
                response = generate_response(query, top_k_results)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.success(response)

# --- Follow-up Question Section ---
if st.session_state.chat_history:
    follow_up = st.text_input("Ask a follow-up question 🧠", key="follow_up_q")

    if st.button("💬 Ask"):
        if not follow_up.strip():
            st.warning("Please enter your follow-up question.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": follow_up})
            with st.spinner("Generating follow-up answer..."):
                # You can modify generate_response to accept chat history
                response = generate_response(follow_up, top_k_results)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.info(response)
# --- Chat History Display ---
if st.session_state.chat_history:
    st.subheader("🗂 Chat History")
    for msg in st.session_state.chat_history:
        role = "🧑‍💻 You" if msg["role"] == "user" else "🤖 Bot"
        st.markdown(f"{role}:** {msg['content']}")

