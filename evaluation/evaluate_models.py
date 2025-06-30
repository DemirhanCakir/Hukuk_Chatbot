# evaluate_llm_models.py
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from sklearn.model_selection import train_test_split
from evaluate import load


load_dotenv()

# PDF veri kaynağını yükle
pdf_path = os.path.abspath("../app/data/gerekceli_anayasa.pdf")
loader = PyPDFLoader(pdf_path)
docs = loader.load()
# Eğitim ve test verisi olarak ayır
train_docs, test_docs = train_test_split(docs, test_size=0.2, random_state=42)
splitter = RecursiveCharacterTextSplitter(chunk_size=800)
chunks = splitter.split_documents(train_docs)
sample_docs = random.sample(chunks, 50)
sample_questions = [train_docs.page_content[:200] + " Bu metne göre anayasal haklar nelerdir?" for train_docs in sample_docs]
reference_answers = [train_docs.page_content.strip()[:500] for train_docs in sample_docs]  # max 500 karakterlik referans

# Prompt tanımı
prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen Türkiye Anayasası hakkında bilgi veren uzman bir asistan botsun. {context}"),
    ("human", "{input}")
])

# Embed ve retriever ayarları
openai_embed = OpenAIEmbeddings(model="text-embedding-3-large")
gemini_embed = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
retriever_gpt = Chroma(embedding_function=openai_embed, persist_directory="../chroma_db_gpt").as_retriever()
retriever_gemini = Chroma(embedding_function=gemini_embed, persist_directory="../chroma_db_gemini").as_retriever()

# LLM modelleri
llm_gpt = ChatOpenAI(model="gpt-4o", temperature=0.3, max_tokens=500)
llm_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)

# Yanıt alma fonksiyonu
def get_answer(llm, retriever, q):
    try:
        qa_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
        return qa_chain.invoke({"input": q})["answer"]
    except Exception as e:
        return ""

# Yanıtları al
gpt_answers = [get_answer(llm_gpt, retriever_gpt, q) for q in sample_questions]
gemini_answers = [get_answer(llm_gemini, retriever_gemini, q) for q in sample_questions]

# BERTScore hesapla
bertscore = load("bertscore")
references = [[r] for r in reference_answers]

scores_gpt = bertscore.compute(predictions=gpt_answers, references=references, lang="tr")
scores_gemini = bertscore.compute(predictions=gemini_answers, references=references, lang="tr")

# Ortalama skorlar
def print_scores(name, scores):
    print(f"\n{name} BERTScore Sonuçları:")
    print(f"F1: {sum(scores['f1']) / len(scores['f1']):.4f}")
    print(f"Precision: {sum(scores['precision']) / len(scores['precision']):.4f}")
    print(f"Recall: {sum(scores['recall']) / len(scores['recall']):.4f}")

print_scores("GPT-4o", scores_gpt)
print_scores("Gemini-1.5-pro", scores_gemini)
