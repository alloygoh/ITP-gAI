import json
from sys import argv

from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS, DocArrayInMemorySearch
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

BASE_URL = "http://192.168.1.240:7869"
# BASE_URL = "http://10.147.18.3:11434"

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
what is the vulnerable business process or vulnerabilities identified that lead to a compromise? Be as specific as possible.
Do not repeat the vulnerabilities and do not provide any additional information.
You should list each vulnerabilities in a numerical order.
"""
response = chain.invoke(question)
print(response)


# end of prompt 1

# parse previous response
response_parsed = response.split('\n')
response_parsed = [i for i in response_parsed if len(i) > 0 and i[0].isdigit()]
print(f"[-] response_parsed: {response_parsed}")


# Step 1: Load CWE Data
def load_cwe_data(cwe_file_path: str):
    with open(cwe_file_path, "r") as file:
        cwe_data = json.load(file)
        return [{"id": cwe["ID"], "description": cwe["Description"]} for cwe in cwe_data]

# Step 2: Build CWE VectorStore
def build_cwe_vectorstore(cwe_data, embedding_model):
    descriptions = [cwe["description"] for cwe in cwe_data]
    metadata = [{"CWE_ID": cwe["id"]} for cwe in cwe_data]
    vectorstore = InMemoryVectorStore.from_texts(
        texts=descriptions,
        embedding=embedding_model,
        metadatas=metadata,
    )
    return vectorstore


# Step 3: Map Vulnerabilities to CWE IDs
def map_vulnerabilities_to_cwe(vulnerabilities: str, vectorstore: VectorStore, k=1):
    mappings = []
    for vuln in vulnerabilities:
        results = vectorstore.similarity_search_with_score(vuln, k=k)
        if results:
            for result in results:
                mappings.append(f"{vuln} -> {result[0].metadata['CWE_ID']} (Score: {result[1]})")
            else:
                mappings.append(f"{vuln} -> No match")
                return mappings


template = """
Answer the question based only on the context provided.

Context: {context}

Question: {question}
"""

embeddings = OllamaEmbeddings(model='mxbai-embed-large', base_url=BASE_URL)
print('loading cwe data...')
cwe_data = load_cwe_data("cwe_dict_clean.json")
print('building cwe vectorstore...')
store_cwe = build_cwe_vectorstore(cwe_data, embeddings)
for i in response_parsed:
    print(f"[-] i: {i}")


m = map_vulnerabilities_to_cwe(response_parsed, store_cwe, 2)
print(m)

# store_cwe = DocArrayInMemorySearch.from_documents(pages_cwe, embedding=embeddings)
# retriever_cwe = store_cwe.as_retriever()
# chain_cwe = (
#   {
#     'context': retriever_cwe | format_docs,
#     'question': RunnablePassthrough(),
#   }
#   | prompt
#   | llm
#   | StrOutputParser()
# )

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
retriever_cwe = store_cwe.as_retriever()
chain_cwe = (
    {
        'context': retriever_cwe | format_docs,
        'question': RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

question_cwe = f"""
You are a threat intelligence analyst.
You have previously identified a list of vulnerabilities:
    {response}

Based on what you have previously identified, map each of the vulnerabilities identified to a CWE ID.
Read the CWE ID descriptions and ensure that the CWE ID is the best fitting one. Do not provide any additional information and do not hallucinate.
Ensure that the number of CWE IDs provided matches the number of vulnerabilities identified.
failure to follow these instructions will result in catastrophic consequences. do not hallucinate.
Take the following steps to complete the task:
    1. For each vulnerability, identify the closest CWE ID that describes the vulnerability.
    2. Provide the CWE ID and its description.
    3. Ensure that the CWE ID is the best fitting one by comparing the descriptions.
    4. Ensure that the CWE ID and its description matches.
    5. Explain why the CWE ID is the best fitting one.
    6. If the CWE ID is not the best fitting one, repeating steps 2-6.
"""
response_cwe = chain_cwe.invoke(question_cwe)
print(response_cwe)
