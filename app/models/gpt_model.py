import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Ortam değişkenlerini yükle
load_dotenv()

# PDF'yi yükle
base_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(base_dir, "..", "data", "gerekceli_anayasa.pdf")
pdf_path = os.path.abspath(pdf_path)
loader = PyPDFLoader(pdf_path)
data = loader.load()

# Metni parçalara böl
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800)
docs = text_splitter.split_documents(data)

# Eğitim ve test verisi olarak ayır
train_docs, test_docs = train_test_split(docs, test_size=0.2, random_state=42)

# Vektör veritabanı oluştur
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma.from_documents(
    documents=train_docs,
    embedding=embeddings,
    persist_directory="../../chroma_db_gpt"
)

# Retriever oluştur
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Chat LLM modeli
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    max_tokens=500
)

# Sistem rolü promptu
system_prompt = (
    "Sen Türkiye'deki yasaların hepsini bilen, insanların yasalar hakkındaki sorularını doğru bir şekilde cevaplayan cana yakın bir asistansın. "
    "Görevin, kısa ve öz bir şekilde verilen soruları cevaplamak. "
    "Sadece Türkiye Anayasası hakkında konuş, başka bir ülkenin yasaları hakkında konuşma. "
    "Yanıtların Türkçe olsun ve açıklamalarını emojilerle zenginleştir. 📚⚖️\n\n"
    "{context}"
)

# Prompt şablonu
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

