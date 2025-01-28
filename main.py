import json
import logging
import argparse

from dataclasses import dataclass
from os import environ, getenv
from pathlib import Path
from re import MULTILINE, search

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


logger = logging.getLogger("gai")
create_vector_db = getenv("CREATE_VECTOR_DB") == "TRUE"


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


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def load_cwe_data(cwe_file_path: str) -> list[dict[str, str]]:
    with open(cwe_file_path, "r") as file:
        cwe_data = json.load(file)
        for entry in cwe_data:
            for member in entry["members"]:
                del member["Potential_Mitigations"]  # drop mitigation for now
        return cwe_data


def build_cwe_vectorstores(
    cwe_data: list[dict[str, str]], embedding_model: OpenAIEmbeddings
) -> Chroma:
    category_vectorstore = Chroma(
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory="./chroma_db",
        collection_name="cwe_category",
        create_collection_if_not_exists=True,
    )
    # ensure clearing of data
    category_vectorstore.delete_collection()

    category_desc: list[str] = []
    category_metadata: list[dict[str, str]] = []
    cwe_desc: list[str] = []
    cwe_metadata: list[dict[str, str]] = []
    for entry in cwe_data:
        category_cwe_id = entry["category_id"]
        category_desc.append(
            f"{category_cwe_id}: {entry['category_name']}. {entry['summary']}"
        )
        category_metadata.append({"CWE_ID": category_cwe_id})

        cwe_desc = [f"{cwe['ID']}: {cwe['Description']}" for cwe in entry["members"]]
        cwe_metadata = [{"CWE_ID": cwe["ID"]} for cwe in entry["members"]]
        Chroma.from_texts(
            texts=cwe_desc,
            embedding=embedding_model,
            metadatas=cwe_metadata,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory="./chroma_db",
            collection_name=f"{category_cwe_id}-store",
            create_collection_if_not_exists=True,
        )

    category_vectorstore = Chroma.from_texts(
        texts=category_desc,
        metadatas=category_metadata,
        embedding=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory="./chroma_db",
        collection_name="cwe_category",
        create_collection_if_not_exists=True,
    )
    return category_vectorstore


def load_cwe_vectorstores(embedding: OpenAIEmbeddings):
    category_vectorstore = Chroma(
        embedding_function=embedding,
        persist_directory="./chroma_db",
        collection_name="cwe_category",
        create_collection_if_not_exists=False,
    )
    if len(category_vectorstore.get()["ids"]) == 0:
        raise InvalidCollectionException
    return category_vectorstore


def get_cwe_vectorstore(
    embeddings: OpenAIEmbeddings,
) -> Chroma | None:
    if create_vector_db:
        if Path("./chroma_db").exists():
            raise FileExistsError(
                "chroma_db directory already exists. Please delete it first."
            )

        logger.debug("loading cwe JSON data from disk...")
        cwe_data = load_cwe_data("cwe_view_mapping.json")
        logger.debug("cwe data loaded.")
        logger.debug("building cwe vectorstore...")
        return build_cwe_vectorstores(cwe_data, embeddings)
    try:
        return load_cwe_vectorstores(embeddings)
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


def get_members_from_category(category_id: str, embedding: OpenAIEmbeddings):
    try:
        member_vectorstore = Chroma(
            embedding_function=embedding,
            persist_directory="./chroma_db",
            collection_name=f"{category_id}-store",
            create_collection_if_not_exists=False,
        )
    except InvalidCollectionException:
        return None
    return member_vectorstore


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


def process_pdf(pdf_path: str):
    out: list[str] = []
    logger = logging.getLogger("gai")

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
    logger.info("---vulnerabilities identified---")
    logger.info(response)
    logger.info("---model response end---\n\n")
    out.append(response)

    # parse previous response
    response_parsed = response.split("\n")
    # get rid of the numbering
    response_parsed = [
        i.split(".", 1)[1] for i in response_parsed if len(i) > 0 and i[0].isdigit()
    ]

    store_category = get_cwe_vectorstore(embeddings)
    if store_category is None:
        raise FileNotFoundError("CWE vector db not found. Please build the db first.")

    logger.debug("---vulnerabilities parsed---")
    for i in response_parsed:
        logger.debug(f"vuln identified: {i}")

    category_res = map_vulnerabilities_to_cwe(response_parsed, store_category, 3)

    system_prompt_category = """
    Answer the question based only on the context provided.

    Your response should match the following format:
        Vulnerability: <vulnerability>
        CWE ID: <CWE ID>
        CWE Category Description: <description>
        Explanation: <explanation>

    Context: {context}
    """

    for mapping in category_res:
        logger.debug("---vectorstore search results---")
        logger.debug(f"vulnerability: {mapping.vulnerability}")
        logger.debug(f"cwe: {[x for x in mapping.cwe_ids]}")
        logger.debug(f"score: {[x for x in mapping.scores]}\n")
        logger.debug("---vectorstore search results end---")

        retriever = DocArrayInMemorySearch.from_documents(
            mapping.docs, embedding=embeddings
        ).as_retriever()

        question_category = f"""
        You are a threat intelligence analyst.
        You have previously identified the following vulnerability:
        {mapping.vulnerability}

        Based on the context provided, identify the best fitting CWE category that describes the vulnerability.
        Format your response according to the prompt.
        """
        response_category = prompt_model(
            llm, system_prompt_category, retriever, question_category
        )

        logger.info("---category identification---")
        logger.info(response_category)
        logger.info("---model response end---\n")
        out.append(response_category)

        extracted = search(r"CWE ID: (.*)\n", response_category, MULTILINE)
        if extracted is None:
            raise ValueError("unable to extract CWE category from model's response.")

        cwe_category = extracted.group(1).strip()

        # now get top 3 CWEs from the category
        members_retriever = get_members_from_category(cwe_category, embeddings)
        if members_retriever is None:
            raise FileNotFoundError(
                "CWE store corresponding to the category not found."
            )

        top_n_cwes = map_vulnerabilities_to_cwe(
            [mapping.vulnerability], members_retriever, 6
        )[0]

        logger.debug("---vectorstore search results---")
        logger.debug(f"cwe: {[x for x in top_n_cwes.cwe_ids]}")
        logger.debug(f"score: {[x for x in top_n_cwes.scores]}\n")
        logger.debug("---vectorstore search results end---\n")

        cwe_retriever = DocArrayInMemorySearch.from_documents(
            top_n_cwes.docs, embedding=embeddings
        ).as_retriever()

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

        question_cwe = f"""
        You are a threat intelligence analyst.
        You have previously identified the following vulnerability:
        {mapping.vulnerability}

        Based on the context provided, rank the CWE IDs that best describes the vulnerability.
        You should only list the top 3 CWE IDs.
        Include the identified vulnerability in the response under "Vulnerability Identified".
        Format your response according to the prompt.
        """

        response_cwe = prompt_model(llm, system_prompt_cwe, cwe_retriever, question_cwe)
        logger.info("---CWE ranking---")
        logger.info(response_cwe)
        logger.info("---model response end---\n")
        out.append(response_cwe)
        return out


def eval_mode(pdf_paths: str):
    pdfs = Path(pdf_paths).rglob("*.pdf")

    for pdf in pdfs:
        for i in range(1, 3):
            out_path = Path(pdf_paths) / f"{pdf.name[:-4]}_0{i}.log"
            pdf_result = process_pdf(pdf.as_posix())
            with open(out_path, "w") as f:
                f.write("\n".join(pdf_result))


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
