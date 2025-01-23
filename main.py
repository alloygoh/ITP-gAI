from dataclasses import dataclass
import json
from os import environ, getenv
from sys import argv

from chromadb.errors import InvalidCollectionException
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


@dataclass
class CweMapping:
    vulnerability: str
    cwe_ids: list[str]
    scores: list[float]
    docs: list[Document]


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def load_cwe_data(cwe_file_path: str) -> list[dict[str, str]]:
    with open(cwe_file_path, "r") as file:
        cwe_data = json.load(file)
        return [
            {"id": cwe["ID"], "description": cwe["Description"]} for cwe in cwe_data
        ]


def build_cwe_vectorstore(
    cwe_data: list[dict[str, str]], embedding_model: OpenAIEmbeddings
) -> Chroma:
    descriptions = [f"{cwe['id']}: {cwe['description']}" for cwe in cwe_data]
    metadata = [{"CWE_ID": cwe["id"]} for cwe in cwe_data]

    vectorstore = Chroma.from_texts(
        texts=descriptions,
        embedding=embedding_model,
        metadatas=metadata,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory="./chroma_db",
        collection_name="cwe",
    )
    return vectorstore


def load_cwe_vectorstore(embedding: OpenAIEmbeddings):
    vectorstore = Chroma(
        embedding_function=embedding,
        persist_directory="./chroma_db",
        collection_name="cwe",
        create_collection_if_not_exists=False,
    )
    if len(vectorstore.get()["ids"]) == 0:
        raise InvalidCollectionException
    return vectorstore


def get_cwe_vectorstore(embeddings: OpenAIEmbeddings) -> Chroma | None:
    if create_vector_db:
        print("loading cwe data...")
        cwe_data = load_cwe_data("cwe_dict_clean.json")
        print("building cwe vectorstore...")
        return build_cwe_vectorstore(cwe_data, embeddings)
    try:
        return load_cwe_vectorstore(embeddings)
    except InvalidCollectionException:
        return None


def map_vulnerabilities_to_cwe(
    vulnerabilities: list[str], vectorstore: VectorStore, k: int = 1
) -> list[CweMapping]:
    mapping_results: list[CweMapping] = []
    for vuln in vulnerabilities:
        results = vectorstore.similarity_search_with_score(vuln, k=k)
        current_mapping = CweMapping(vulnerability=vuln, cwe_ids=[], docs=[], scores=[])
        if results:
            for result in results:
                current_mapping.cwe_ids.append(result[0].metadata["CWE_ID"])
                current_mapping.scores.append(result[1])
                current_mapping.docs.append(result[0])
        mapping_results.append(current_mapping)
    return mapping_results


# load env vars from .env file
load_dotenv()

create_vector_db = getenv("CREATE_VECTOR_DB") == "TRUE"

if "OPENAI_API_KEY" not in environ:
    print("Please set the OPENAI_API_KEY environment variable.")
    exit()

llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

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
if the vulnerability stems from outdated software, state that it is a dependency on a vulnerable third-party component.
Do not repeat the vulnerabilities and do not provide any additional information.
You should list each vulnerabilities in a numerical order.
"""
response = rag_chain.invoke({"input": question})
print(response["answer"])


# parse previous response
response_parsed = response["answer"].split("\n")
# get
response_parsed = [
    i.split(".", 1)[1] for i in response_parsed if len(i) > 0 and i[0].isdigit()
]
print(f"[-] response_parsed: {response_parsed}")

# embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url=BASE_URL)
store_cwe = get_cwe_vectorstore(embeddings)
if store_cwe is None:
    print("CWE vector db not found. Please build the db first.")
    exit()

for i in response_parsed:
    print(f"[-] vuln identified: {i}")

res = map_vulnerabilities_to_cwe(response_parsed, store_cwe, 3)

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

for mapping in res:
    print("--------------------")
    print(f"[-] vulnerability: {mapping.vulnerability}")
    print(f"[-] cwe: {[x for x in mapping.cwe_ids]}")
    print(f"[-] score: {[x for x in mapping.scores]}\n")
    # chain of operations
    qa_chain_cwe = create_stuff_documents_chain(llm, cwe_prompt)
    retriever_cwe = DocArrayInMemorySearch.from_documents(
        mapping.docs, embedding=embeddings
    ).as_retriever()
    rag_chain_cwe = create_retrieval_chain(retriever_cwe, qa_chain_cwe)

    question_cwe = f"""
    You are a threat intelligence analyst.
    You have previously identified the following vulnerability:
    {mapping.vulnerability}

    Based on the context provided, identify the best fitting CWE ID that describes the vulnerability.
    Format your response according to the prompt.
    """

    response_cwe = rag_chain_cwe.invoke({"input": question_cwe})
    print(f"response from model:\n{response_cwe['answer']}")
    print("--------------------\n\n")

# this does not work as well, prone to some hallucination
# question_cwe = f"""
# You are a threat intelligence analyst.
# You have previously identified a list of vulnerabilities:
#     {response}

# Based on what you have previously identified, map each of the vulnerabilities identified to a CWE ID.
# Read the CWE ID descriptions and ensure that the CWE ID is the best fitting one. Do not provide any additional information and do not hallucinate.
# Ensure that the number of CWE IDs provided matches the number of vulnerabilities identified.
# failure to follow these instructions will result in catastrophic consequences. do not hallucinate.
# Take the following steps to complete the task:
#     1. For each vulnerability, list the vulnerability.
#     2. Identify the best fitting CWE ID that describes the vulnerability by comparing the CWE description with the vulnerability.
#     3. Provide the CWE ID and its description.
#     4. Check that the CWE ID and its description matches.
#     5. Explain why the CWE ID is the best fitting one.
#     6. Ensure that the CWE ID is the best fitting one by comparing the description of the CWE and the vulnerability.
#     7. If the CWE ID is not the best fitting one, repeating steps 2-7.
# """


# relevant_docs = []
# for mapping in res:
#     relevant_docs.extend(mapping.docs)

# qa_chain_cwe = create_stuff_documents_chain(llm, cwe_prompt)
# retriever_cwe = DocArrayInMemorySearch.from_documents(
#     relevant_docs, embedding=embeddings
# ).as_retriever()
# rag_chain_cwe = create_retrieval_chain(retriever_cwe, qa_chain_cwe)
# response_cwe = rag_chain_cwe.invoke({"input": question_cwe})
# print(response_cwe["answer"])
