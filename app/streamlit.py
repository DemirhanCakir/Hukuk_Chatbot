# streamlit.py
import streamlit as st
from dotenv import load_dotenv
from models.gpt_model import retriever as gpt_retriever, llm as gpt_llm, prompt as gpt_prompt
from models.gemini_model import retriever as gemini_retriever, llm as gemini_llm, prompt_template as gemini_prompt
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

load_dotenv()

# Sayfa ayarları
st.set_page_config(page_title="Anayasa Asistanı", layout="wide")
st.title("📜 Türkiye Anayasası Sohbet Asistanı")

# Sidebar model seçimi
model_choice = st.sidebar.selectbox(
    "🔍 Lütfen bir model seçin:",
    ("gpt-4o", "gemini-1.5-pro")
)

# Model yapılandırması
if model_choice == "gpt-4o":
    retriever = gpt_retriever
    llm = gpt_llm
    prompt = gpt_prompt
else:
    retriever = gemini_retriever
    llm = gemini_llm
    prompt = gemini_prompt

rag_chain = create_retrieval_chain(
    retriever,
    create_stuff_documents_chain(llm, prompt)
)

# Session state'de mesaj geçmişini tut
if "messages" not in st.session_state:
    st.session_state.messages = []

# Önceki mesajları göster
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Yeni mesaj girişi
user_input = st.chat_input("Sorunuzu yazın...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = rag_chain.invoke({"input": user_input})
        answer = response["answer"]
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
