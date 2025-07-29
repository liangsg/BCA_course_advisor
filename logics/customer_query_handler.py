
import json
from helper_functions import llm
from bs4 import BeautifulSoup
import requests
from langchain_chroma import Chroma
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

#step1 Document Loading
file_path = 'data/specialist_diploma_programmes_urls.json'
# Load JSON data into a list
with open(file_path, 'r', encoding='utf-8') as f:
    course_data = json.load(f)

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    # Remove scripts/styles and extract visible text
    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()
    return soup.get_text(separator=" ", strip=True)

texts = [extract_text_from_url(item["url"]) for item in course_data]

#step2 Splitting and Chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
documents = []

for i, text in enumerate(texts):
    url = course_data[i]["url"]
    doc_chunks = splitter.create_documents([text], metadatas=[{"source": url}])
    documents.extend(doc_chunks)

#step3 Embedding
#step4 Store in Chroma Vector Store
vector_store = Chroma.from_documents(
    collection_name="course_data",
    documents=documents,
    embedding=llm.embedding,
)
#print(vector_store._collection.count())
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

#step5 Retrieval with Crossencoder
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
prompt = PromptTemplate.from_template("""
You are Course Advisory Assistant for Singapore Building and Construction Authority (BCA) Academy Specialist Diploma courses
Use the following context to answer the question.
If the answer is not in the context, say "I don't know."

{context}

Question: {question}
Answer:""")

qa_chain = LLMChain(llm=llm.llm_QA, prompt=prompt)

#Reranker function
def rerank_documents_crossencoder(query: str, docs: list[Document], top_n: int = 3):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_n]]

#Custom_qa function with reranking
def custom_qa_with_rerank(query: str, top_k_retrieve: int = 10, top_k_rerank: int = 3):
    retrieved_docs = retriever.get_relevant_documents(query)
    reranked_docs = rerank_documents_crossencoder(query, retrieved_docs, top_n=top_k_rerank)
    context = "\n\n".join([doc.page_content for doc in reranked_docs])
    answer = qa_chain.invoke({"context": context, "question": query})
    return {"result": answer, "source_documents": reranked_docs}
