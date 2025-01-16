from sys import argv

from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import DocArrayInMemorySearch, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.llms import OllamaLLM

BASE_URL = "http://10.147.18.3:11434"

def format_docs(docs : list[Document]) -> str:
  return "\n\n".join(doc.page_content for doc in docs)

def format_string(string : str) -> str:
  return string

llm = OllamaLLM(model='llama3.2:1B', base_url=BASE_URL)
embeddings = OllamaEmbeddings(model='bge-large', base_url=BASE_URL)
# llm = OllamaLLM(model='llama3.1', base_url=BASE_URL)
# embeddings = OllamaEmbeddings(model='znbang/bge:large-en-v1.5-f16', base_url=BASE_URL)

# load PDF
# TODO: find better parsers and tokeniser or what not
loader = PyPDFLoader(argv[1])
pages = loader.load_and_split()
store = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
retriever = store.as_retriever()

template = """
Answer the question based only on the context provided.

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)


# chain of operations
chain = (
  {
    'context': retriever | format_docs,
    'question': RunnablePassthrough(),
  }
  | prompt
  | llm
  | StrOutputParser()
)

question = """
You are a threat intelligence analyst. look only at the "facts of this case" section.
what is the vulnerable business process or vulnerabilities identified that lead to a compromise? Only list the vulnerabilties and be as specific as possible.
Do not repeat the vulnerabilities and do not provide any additional information.
"""
response = chain.invoke(question)
print(response)



template = """
Answer the question based only on the context provided.

Context: {context}

Question: {question}
"""

llm = OllamaLLM(model='llama3.2:1B', base_url=BASE_URL)
embeddings = OllamaEmbeddings(model='llama3.2:1B', base_url=BASE_URL)
prompt_cwe = PromptTemplate.from_template(template)
loader_cwe = PyPDFLoader('cwe.pdf')
pages_cwe = loader_cwe.load_and_split()
texts = []
for p in pages_cwe:
    p.page_content = p.page_content.replace('.', '')
    texts.append(p.page_content)

print(texts)
store_cwe = DocArrayInMemorySearch.from_texts(texts, embedding=embeddings)
# store_cwe = DocArrayInMemorySearch.from_documents(pages_cwe, embedding=embeddings)
retriever_cwe = store.as_retriever()
chain_cwe = (
  {
    'context': retriever_cwe | format_docs,
    'question': RunnablePassthrough(),
  }
  | prompt
  | llm
  | StrOutputParser()
)

# db = FAISS.from_documents(pages_cwe, embedding=embeddings)
# store_cwe = InMemoryVectorStore.from_documents(pages_cwe, embedding=embeddings)
# docs_store: list[Document] = []
# for i in response.split('\n'):
#   print(f"[-] i: {i}")
#   q = f'What is the closest CWE ID for the following vulnerability: {i}'
#   # docs = store_cwe.similarity_search(i)
#   docs = db.similarity_search(i)
#   print("[+] listing relevant documents:")
#   for doc in docs:
#     print(doc.page_content)
#   docs_store.extend(docs)
# print("[+] relevant documents loaded")

# retriever_cwe = DocArrayInMemorySearch.from_documents(docs_store, embedding=embeddings).as_retriever()
# chain_cwe = (
#   {
#     'reference': retriever_cwe | format_docs,
#     'question': RunnablePassthrough(),
#   }
#   | prompt_cwe
#   | llm
#   | StrOutputParser()
# )

question_cwe = """
You are a threat intelligence analyst. Based on what you have previously identified, map each of the vulnerabilities identified to a CWE ID.
Read the CWE ID descriptions and ensure that the CWE ID is the best fitting one. Do not provide any additional information and do not hallucinate.
ensure that the number of vulnerabilities identified matches the number of CWE IDs provided.
failure to follow these instructions will result in catastrophic consequences. do not hallucinate.
"""
response_cwe = chain_cwe.invoke(question_cwe)
print(response_cwe)
