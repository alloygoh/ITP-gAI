import json
from os import environ, getenv
from sys import argv

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# load env vars from .env file
load_dotenv()

hf_token = getenv("HUGGINGFACE_TOKEN")

if "OPENAI_API_KEY" not in environ:
    print("Please set the OPENAI_API_KEY|HUGGINGFACE_TOKEN environment variable.")
    exit()

llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
# embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url=BASE_URL)


# load PDF
# TODO: find better parsers and tokeniser or what not
loader = PyPDFLoader(argv[1])
pages = loader.load_and_split()
store = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
retriever = store.as_retriever()

system_prompt = """
Answer the question based only on the context provided.

Context: {context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# chain of operations
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# You are a threat intelligence analyst. look only at the "facts of this case" and "Findings and Basis for Determination" sections.
question = """
You are a threat intelligence analyst. Look at the breach report provided.
what are the vulnerable business process or vulnerabilities identified that lead to a compromise? Be as specific as possible.
You should state each vulnerability in a sentence, and provide a short description not exceeding 1 sentence.
Do not repeat the vulnerabilities and do not provide any additional information.
You should list each vulnerabilities in a numerical order.
"""
response = rag_chain.invoke({"input": question})
print(response["answer"])


# end of prompt 1

# parse previous response
response_parsed = response["answer"].split("\n")
response_parsed = [i for i in response_parsed if len(i) > 0 and i[0].isdigit()]
print(f"[-] response_parsed: {response_parsed}")


# Step 1: Load CWE Data
def load_cwe_data(cwe_file_path: str):
    with open(cwe_file_path, "r") as file:
        cwe_data = json.load(file)
        return [
            {"id": cwe["ID"], "description": cwe["Description"]} for cwe in cwe_data
        ]


# Step 2: Build CWE VectorStore
def build_cwe_vectorstore(cwe_data, embedding_model):
    descriptions = [f"{cwe['id']}: {cwe['description']}" for cwe in cwe_data]
    metadata = [{"CWE_ID": cwe["id"]} for cwe in cwe_data]

    vectorstore = Chroma.from_texts(
        texts=descriptions,
        embedding=embedding_model,
        metadatas=metadata,
        collection_metadata={"hnsw:space": "cosine"},
    )
    # vectorstore = InMemoryVectorStore.from_texts(
    #     texts=descriptions,
    #     embedding=embedding_model,
    #     metadatas=metadata,
    # )
    return vectorstore


# Step 3: Map Vulnerabilities to CWE IDs
def map_vulnerabilities_to_cwe(
    vulnerabilities: list[str], vectorstore: VectorStore, k=1
):
    mappings = []
    docs = []
    for vuln in vulnerabilities:
        results = vectorstore.similarity_search_with_score(vuln, k=k)
        if results:
            for result in results:
                mappings.append(
                    f"{vuln} -> {result[0].metadata['CWE_ID']} (Score: {result[1]})"
                )
                docs.append(result[0])
        else:
            mappings.append(f"{vuln} -> No match")
    return mappings, docs


# embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url=BASE_URL)
print("loading cwe data...")
cwe_data = load_cwe_data("cwe_dict_clean.json")
print("building cwe vectorstore...")
store_cwe = build_cwe_vectorstore(cwe_data, embeddings)
for i in response_parsed:
    print(f"[-] i: {i}")

m, out = map_vulnerabilities_to_cwe(response_parsed, store_cwe, 3)
for i in m:
    print(f"[+] i: {i}")

system_prompt_cwe = """
Answer the question based only on the context provided.

Your response should match the following format:
    Vulnerability: <vulnerability>
    CWE ID: <CWE ID>
    Description: <description>
    Explanation: <explanation>

Context: {context}
"""

cwe_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_cwe),
        ("human", "{input}"),
    ]
)

# chain of operations
qa_chain_cwe = create_stuff_documents_chain(llm, cwe_prompt)
retriever_cwe = DocArrayInMemorySearch.from_documents(
    out, embedding=embeddings
).as_retriever()
# retriever_cwe = store_cwe.as_retriever()
rag_chain_cwe = create_retrieval_chain(retriever_cwe, qa_chain_cwe)

question_cwe = f"""
You are a threat intelligence analyst.
You have previously identified a list of vulnerabilities:
{response["answer"]}

Based on what you have previously identified, map each of the vulnerabilities identified to a CWE ID.

Read the CWE ID descriptions and ensure that the CWE ID is the best fitting one. Do not provide any additional information and do not hallucinate.
Ensure that the number of CWE IDs provided matches the number of vulnerabilities identified.
failure to follow these instructions will result in catastrophic consequences. do not hallucinate.
Take the following steps to complete the task:
    1. For each vulnerability, list the vulnerability.
    2. Identify the best fitting CWE ID that describes the vulnerability by comparing the CWE description with the vulnerability.
    3. Provide the CWE ID and its description.
    4. Check that the CWE ID and its description matches.
    5. Explain why the CWE ID is the best fitting one.
    6. Ensure that the CWE ID is the best fitting one by comparing the description of the CWE and the vulnerability.
    7. If the CWE ID is not the best fitting one, repeating steps 2-7.
Ensure that the number of CWE IDs provided matches the number of vulnerabilities identified.
Format your response according to the prompt.
"""

print(question_cwe)

response_cwe = rag_chain_cwe.invoke({"input": question})
print(response_cwe["answer"])
