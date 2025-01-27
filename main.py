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
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
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
        for entry in cwe_data:
            for member in entry["members"]:
                del member["Potential_Mitigations"]  # drop mitigation for now
        return cwe_data
        # return [
        #     {"id": cwe["ID"], "description": cwe["Description"]} for cwe in cwe_data
        # ]


def build_cwe_vectorstores(
    cwe_data: list[dict[str, str]], embedding_model: OpenAIEmbeddings
) -> tuple[Chroma, Chroma]:
    cwe_vectorstore = Chroma(
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory="./chroma_db",
        collection_name="cwe",
        create_collection_if_not_exists=True,
    )

    category_vectorstore = Chroma(
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory="./chroma_db",
        collection_name="cwe_category",
        create_collection_if_not_exists=True,
    )

    category_desc: list[str] = []
    category_metadata: list[dict[str, str]] = []
    cwe_desc: list[str] = []
    cwe_metadata: list[dict[str, str]] = []
    for entry in cwe_data:
        category_desc.append(
            f"{entry['category_id']}: {entry['category_name']}. {entry['summary']}"
        )
        category_metadata.append({"CWE_ID": entry["category_id"]})

        cwe_desc.extend(
            [f"{cwe['ID']}: {cwe['Description']}" for cwe in entry["members"]]
        )
        cwe_metadata.extend([{"CWE_ID": cwe["ID"]} for cwe in entry["members"]])

    category_vectorstore.add_texts(
        texts=category_desc,
        metadatas=category_metadata,
    )
    cwe_vectorstore.add_texts(
        texts=cwe_desc,
        metadatas=cwe_metadata,
    )

    # descriptions = [f"{cwe['id']}: {cwe['description']}" for cwe in cwe_data]
    # metadata = [{"CWE_ID": cwe["id"]} for cwe in cwe_data]

    # vectorstore = Chroma.from_texts(
    #     texts=descriptions,
    #     embedding=embedding_model,
    #     metadatas=metadata,
    #     collection_metadata={"hnsw:space": "cosine"},
    #     persist_directory="./chroma_db",
    #     collection_name="cwe",
    # )
    return category_vectorstore, cwe_vectorstore


def load_cwe_vectorstores(embedding: OpenAIEmbeddings):
    cwe_vectorstore = Chroma(
        embedding_function=embedding,
        persist_directory="./chroma_db",
        collection_name="cwe",
        create_collection_if_not_exists=False,
    )
    category_vectorstore = Chroma(
        embedding_function=embedding,
        persist_directory="./chroma_db",
        collection_name="cwe_category",
        create_collection_if_not_exists=False,
    )
    if (
        len(cwe_vectorstore.get()["ids"]) == 0
        or len(category_vectorstore.get()["ids"]) == 0
    ):
        raise InvalidCollectionException
    return category_vectorstore, cwe_vectorstore


def get_cwe_vectorstore(
    embeddings: OpenAIEmbeddings,
) -> tuple[Chroma, Chroma] | tuple[None, None]:
    if create_vector_db:
        print("loading cwe data...")
        cwe_data = load_cwe_data("cwe_view_mapping.json")
        print("building cwe vectorstore...")
        return build_cwe_vectorstores(cwe_data, embeddings)
    try:
        return load_cwe_vectorstores(embeddings)
    except InvalidCollectionException:
        return None, None


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


def prompt_model(
    llm: ChatOpenAI,
    system_prompt: str,
    retriever: VectorStoreRetriever,
    question: str,
):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    # chain of operations
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    response = rag_chain.invoke({"input": question})
    return response["answer"]


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
Do not repeat yourself.

Context: {context}
"""

# You are a threat intelligence analyst. look only at the "facts of this case" and "Findings and Basis for Determination" sections.
question = """
You are a threat intelligence analyst. Look at the breach report provided.
what are the vulnerable business process or vulnerabilities identified that lead to a compromise? Be as specific as possible.
You should state each vulnerability in a sentence, and provide a short description not exceeding 2 sentences.
if the vulnerability stems from outdated software or a bug in an external software, state that it is a dependency on a vulnerable third-party component and a violation of Secure Design Principles.
Do not repeat the vulnerabilities and do not provide any additional information.
If the identified vulnerability is an in-depth version of a previously identified vulnerability, ignore it and move on.
You should list each vulnerability in a numerical order.
"""
response = prompt_model(llm, system_prompt, retriever, question)
print(response)


# parse previous response
response_parsed = response.split("\n")
# get rid of the numbering
response_parsed = [
    i.split(".", 1)[1] for i in response_parsed if len(i) > 0 and i[0].isdigit()
]
print(f"[-] response_parsed: {response_parsed}")

store_category, store_cwe = get_cwe_vectorstore(embeddings)
if store_cwe is None or store_category is None:
    print("CWE vector db not found. Please build the db first.")
    exit()

for i in response_parsed:
    print(f"[-] vuln identified: {i}")

category_res = map_vulnerabilities_to_cwe(response_parsed, store_category, 3)

system_prompt_cwe = """
Answer the question based only on the context provided.

Your response should match the following format:
    Vulnerability: <vulnerability>
    CWE ID: <CWE ID>
    CWE Category Description: <description>
    Explanation: <explanation>

Context: {context}
"""

for mapping in category_res:
    print("--------------------")
    print(f"[-] vulnerability: {mapping.vulnerability}")
    print(f"[-] cwe: {[x for x in mapping.cwe_ids]}")
    print(f"[-] score: {[x for x in mapping.scores]}\n")
    retriever = DocArrayInMemorySearch.from_documents(
        mapping.docs, embedding=embeddings
    ).as_retriever()

    question_cwe = f"""
    You are a threat intelligence analyst.
    You have previously identified the following vulnerability:
    {mapping.vulnerability}

    Based on the context provided, identify the best fitting CWE category that describes the vulnerability.
    Format your response according to the prompt.
    """
    response_cwe = prompt_model(llm, system_prompt_cwe, retriever, question_cwe)

    # response_cwe = rag_chain_cwe.invoke({"input": question_cwe})
    print(f"response from model:\n{response_cwe}")
    print("--------------------\n\n")
    break
