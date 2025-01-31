from dataclasses import dataclass
import json
from os import environ, getenv
import argparse
from pathlib import Path
import logging

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


def process_pdf(pdf_path: str):
    out: list[str] = []
    llm = ChatOpenAI(model="gpt-4o")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # load PDF
    # TODO: find better parsers and tokeniser or what not
    loader = PyPDFLoader(pdf_path)
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
    You should state each vulnerability in a sentence, and provide a short description not exceeding 2 sentences.
    if the vulnerability stems from outdated software or a bug in an external software, state that it is a dependency on a vulnerable third-party component and a violation of Secure Design Principles in your explanation.
    Do not repeat the vulnerabilities and do not provide any additional information.
    If the identified vulnerability is an in-depth version of a previously identified vulnerability, ignore it and move on.
    If the vulnerability is reasonably inferred from a previous vulnerability, ignore it and move on.
    Focus on unique vulnerabilities that have not been previously identified.
    You should list each vulnerability in a numerical order.
    """
    response = rag_chain.invoke({"input": question})
    print(response["answer"])
    out.append(response["answer"])

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
        Ranking: <ranking>
        Vulnerability Identified: <vulnerability>
        CWE ID: <CWE ID>
        CWE Description: <description>
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

        Based on the context provided, rank the CWE IDs that best describes the vulnerability.
        You should only list the top 3 CWE IDs.
        Your decision should be informed by the technical details, keywords, and context provided.
        Include the identified vulnerability in the response under "Vulnerability Identified".
        Format your response according to the prompt.
        """

        response_cwe = rag_chain_cwe.invoke({"input": question_cwe})
        print(f"response from model:\n{response_cwe['answer']}")
        print("--------------------\n\n")
        out.append(response_cwe["answer"])
    return out


def eval_mode(pdf_paths: str):
    pdfs = Path(pdf_paths).rglob("*.pdf")

    for pdf in pdfs:
        for i in range(1, 3):
            out_path = Path(pdf_paths) / f"{pdf.name[:-4]}_0{i}.log"
            pdf_result = process_pdf(pdf.as_posix())
            with open(out_path, "w") as f:
                f.write("\n\n".join(pdf_result))


def parse_args():
    parser = argparse.ArgumentParser()

    # Adding the --log argument to set a specific logging level
    parser.add_argument(
        "--log",
        "-l",
        default="INFO",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    parser.add_argument(
        "--eval",
        "-e",
        action="store_true",
        help="Run the program in evaluation mode",
    )

    # accept the path to the PDF file
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file to be processed. If using evaluation mode, this should be a directory containing multiple PDF files.",
    )

    return parser.parse_args()


def main():
    # set up logging to log messages only from this module
    args = parse_args()
    logger = logging.getLogger("gai")
    logger.setLevel(args.log)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(sh)

    # load env vars from .env file
    load_dotenv()

    if "OPENAI_API_KEY" not in environ:
        raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")

    if args.eval:
        eval_mode(args.pdf_path)
        return
    process_pdf(args.pdf_path)


if __name__ == "__main__":
    main()
