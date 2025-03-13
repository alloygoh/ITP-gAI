import argparse
import json
import logging
import re
from dataclasses import dataclass
from os import environ, getenv
from pathlib import Path

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
    impact: str


@dataclass
class ProcessMapping:
    vulnerability: str
    process_ids: list[str]
    scores: list[float]
    docs: list[Document]


# load env vars from .env file
load_dotenv()
logger = logging.getLogger("gai")
create_vector_db = getenv("CREATE_VECTOR_DB") == "TRUE"


def parse_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments including:
            - log (str): Logging level specified by the user (default: "INFO").
            - eval (bool): Flag indicating whether to run the program in evaluation mode.
            - pdf_path (str): Path to the PDF file to be processed or a directory containing multiple PDF files if in evaluation mode.
    """
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


def load_cwe_data(cwe_file_path: str) -> list[dict[str, str]]:
    """
    Load CWE (Common Weakness Enumeration) data from a JSON file.

    Args:
        cwe_file_path (str): The file path to the CWE JSON file.

    Returns:
        list[dict[str, str]]: A list of dictionaries, each containing the "id" and "description" of a CWE.
    """
    with open(cwe_file_path, "r") as file:
        cwe_data = json.load(file)
    return [{"id": cwe["ID"], "description": cwe["Description"]} for cwe in cwe_data]


def load_cwe_mitigations(cwe_file_path: str) -> list[dict[str, str]]:
    """
    Load CWE mitigations from a JSON file.

    Args:
        cwe_file_path (str): The file path to the CWE JSON file.

    Returns:
        list[dict[str, str]]: A list of dictionaries containing CWE IDs and their potential mitigations.
    """
    with open(cwe_file_path, "r") as file:
        cwe_data = json.load(file)
    return [
        {"id": cwe["ID"], "potential_mitigations": cwe["Potential_Mitigations"]}
        for cwe in cwe_data
    ]


def load_process_categories(category_file_path: str) -> list[dict[str, str]]:
    """
    Loads APQC categories from a JSON file.

    Args:
        category_file_path (str): The file path to the JSON file containing category data.

    Returns:
        list[dict[str, str]]: A list of dictionaries where each dictionary represents a category.
    """
    with open(category_file_path, "r") as file:
        category_data = json.load(file)
    return category_data


def build_cwe_vectorstores(
    cwe_data: list[dict[str, str]], embedding_model: OpenAIEmbeddings
) -> Chroma:
    """
    Builds a Chroma vector store from a list of CWE data using the specified embedding model.

    Args:
        cwe_data (list[dict[str, str]]): A list of dictionaries containing CWE data,
                                         where each dictionary has keys 'id' and 'description'.
        embedding_model (OpenAIEmbeddings): The embedding model to use for generating embeddings.

    Returns:
        Chroma: A Chroma vector store containing the embedded CWE descriptions and associated metadata.
    """
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


def build_process_category_vectorstores(
    proc_data: list[dict[str, str]], embedding: OpenAIEmbeddings
) -> Chroma:
    """
    Builds a Chroma vector store from a list of process data dictionaries extracted from the APQC PCF using the specified embedding model.

    Args:
        proc_data (list[dict[str, str]]): A list of dictionaries containing process data.
            Each dictionary should have keys "ID" and "Description".
        embedding (OpenAIEmbeddings): The embedding model to use for generating embeddings.

    Returns:
        Chroma: A Chroma vector store containing the embedded process descriptions and their metadata.
    """
    descriptions = [f"{proc['ID']}: {proc['Description']}" for proc in proc_data]
    metadata = [{"ID": proc["ID"]} for proc in proc_data]

    vectorstore = Chroma.from_texts(
        texts=descriptions,
        embedding=embedding,
        metadatas=metadata,
        persist_directory="./chroma_db",
        collection_name="process_categories",
    )
    return vectorstore


def load_cwe_vectorstores(embedding: OpenAIEmbeddings):
    """
    Load the CWE vector store using the provided embedding function.

    This function initializes a Chroma vector store with the specified
    embedding function, under the chroma_db directory, and cwe collection name. If the
    collection does not exist or is empty, an InvalidCollectionException is raised.

    Args:
        embedding (OpenAIEmbeddings): The embedding function to use for the vector store.

    Returns:
        Chroma: The initialized Chroma vector store.

    Raises:
        InvalidCollectionException: If the collection does not exist or is empty.
    """
    vectorstore = Chroma(
        embedding_function=embedding,
        persist_directory="./chroma_db",
        collection_name="cwe",
        create_collection_if_not_exists=False,
    )
    if len(vectorstore.get()["ids"]) == 0:
        raise InvalidCollectionException
    return vectorstore


def load_process_category_vectorstores(embedding: OpenAIEmbeddings):
    """
    Loads and returns a Chroma vector store for APQC process categories.

    This function initializes a Chroma vector store using the provided embedding function.
    It attempts to load the vector store from the chroma_db directory and process_categories collection name.
    If the collection does not exist or is empty, an InvalidCollectionException is raised.

    Args:
        embedding (OpenAIEmbeddings): The embedding function to use for the vector store.

    Returns:
        Chroma: The initialized Chroma vector store.

    Raises:
        InvalidCollectionException: If the collection is empty or does not exist.
    """
    vectorstore = Chroma(
        embedding_function=embedding,
        persist_directory="./chroma_db",
        collection_name="process_categories",
        create_collection_if_not_exists=False,
    )
    if len(vectorstore.get()["ids"]) == 0:
        raise InvalidCollectionException
    return vectorstore


def get_vectorstores(
    embeddings: OpenAIEmbeddings,
) -> tuple[Chroma, Chroma] | tuple[None, None]:
    """
    Retrieves/creates vector stores for the APQC process categories and CWE data.

    If `create_vector_db` is True, this function will create new vector stores
    for process categories and CWE data using the provided embeddings. If the
    `chroma_db` directory already exists, a `FileExistsError` will be raised.

    If `create_vector_db` is False, this function will attempt to load existing
    vector stores from disk. If the vector stores cannot be loaded due to an
    `InvalidCollectionException`, the function will return a tuple of (None, None).

    Args:
        embeddings (OpenAIEmbeddings): The embeddings to use for creating or loading
                                       the vector stores.

    Returns:
        tuple[Chroma, Chroma] | tuple[None, None]: A tuple containing the process
                                                   category vector store and the
                                                   CWE vector store, or (None, None)
                                                   if loading the vector stores fails.
    """
    if create_vector_db:
        if Path("./chroma_db").exists():
            raise FileExistsError(
                "chroma_db directory already exists. Please delete it first."
            )
        logger.debug("loading process category data from disk...")
        proc_data = load_process_categories("process_categories.json")
        logger.debug("process category data loaded.")
        logger.debug("building process category vectorstore...")
        category_vectorstore = build_process_category_vectorstores(
            proc_data, embeddings
        )
        logger.debug("loading cwe JSON data from disk...")
        cwe_data = load_cwe_data("cwe_dict_clean.json")
        logger.debug("cwe data loaded.")
        logger.debug("building cwe vectorstore...")
        cwe_vectorstore = build_cwe_vectorstores(cwe_data, embeddings)
        return category_vectorstore, cwe_vectorstore
    try:
        category_vectorstore = load_process_category_vectorstores(embeddings)
        cwe_vectorstore = load_cwe_vectorstores(embeddings)
        return category_vectorstore, cwe_vectorstore
    except InvalidCollectionException:
        return None, None


def map_vulnerabilities_to_cwe(
    vulnerabilities: list[str],
    vectorstore: VectorStore,
    k: int = 1,
    impacts: list[str] = [],
) -> list[CweMapping]:
    """
    Retrieves top k CWE mappings for a list of vulnerabilities using a vector store for similarity search.
    This function should be used as part of RAG-based operations to increase accuracy of the model's answers.

    Args:
        vulnerabilities (list[str]): A list of vulnerability descriptions.
        vectorstore (VectorStore): An instance of a vector store used for similarity search.
        k (int, optional): The number of top similar results to retrieve for each vulnerability. Defaults to 1.
        impacts (list[str], optional): A list of impact descriptions corresponding to the vulnerabilities. Defaults to an empty list.

    Returns:
        list[CweMapping]: A list of CweMapping objects containing the mapping results for each vulnerability.
    """
    mapping_results: list[CweMapping] = []
    for i, vuln in enumerate(vulnerabilities):
        results = vectorstore.similarity_search_with_score(vuln, k=k)
        impact = ""
        if len(impacts) > i:
            impact = impacts[i]
            impact_results = vectorstore.similarity_search_with_score(impact, k=3)
            results.extend(impact_results)
        current_mapping = CweMapping(
            vulnerability=vuln, cwe_ids=[], docs=[], scores=[], impact=impact
        )
        for result in results:
            current_mapping.cwe_ids.append(result[0].metadata["CWE_ID"])
            current_mapping.scores.append(result[1])
            current_mapping.docs.append(result[0])
        mapping_results.append(current_mapping)
    return mapping_results


def map_vulns_to_processes(
    vulnerabilities: list[str], vectorstore: VectorStore, k: int = 1
) -> list[ProcessMapping]:
    """
    Retrieves top k business processes for a list of vulnerabilities using a vector store for similarity search.
    This function should be used as part of RAG-based operations to increase accuracy of the model's answers.

    Args:
        vulnerabilities (list[str]): A list of vulnerability descriptions.
        vectorstore (VectorStore): An instance of a vector store to perform similarity searches.
        k (int, optional): The number of top similar processes to retrieve for each vulnerability. Defaults to 1.

    Returns:
        list[ProcessMapping]: A list of ProcessMapping objects containing the mapping results.
    """
    mapping_results: list[ProcessMapping] = []
    for vuln in vulnerabilities:
        results = vectorstore.similarity_search_with_score(vuln, k=k)
        current_mapping = ProcessMapping(
            vulnerability=vuln, process_ids=[], docs=[], scores=[]
        )
        for result in results:
            current_mapping.process_ids.append(result[0].metadata["ID"])
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
    """
    Wrapper function to generate a response to a given question using a language model, retriever and system prompt.

    Args:
        llm (ChatOpenAI): The language model to use for generating responses.
        system_prompt (str): The system prompt to provide context to the language model.
        retriever (VectorStoreRetriever): The retriever to use for fetching relevant documents.
        question (str): The question to generate a response for.

    Returns:
        str: The generated response to the question.
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
    response = rag_chain.invoke({"input": question})
    return response["answer"]


def get_relevant_details(response: str):
    """
    Extracts and returns relevant details from a given response string.
    This function should be used to parse the response generated by the language model.

    The function searches for patterns of vulnerabilities, their descriptions,
    and impacts within the response string. It ensures that the number of
    vulnerabilities, descriptions, and impacts are equal. If they are not,
    the function returns None for both details and impacts.

    Args:
        response (str): The response string containing vulnerability details.

    Returns:
        tuple: A tuple containing two elements:
            - details (list[str]): A list of strings where each string is a
              combination of a vulnerability and its corresponding description.
            - impacts (list[str]): A list of impact strings corresponding to
              each vulnerability.
    """
    vulns = re.findall(r"Vulnerability: (.+)", response)
    descriptions = re.findall(r"Explanation: (.+)", response)
    impacts = re.findall(r"Impact: (.+)", response)
    if not (len(vulns) == len(descriptions) == len(impacts)):
        return None, None
    details: list[str] = []
    for i in range(len(vulns)):
        details.append(f"{vulns[i]}: {descriptions[i]}")
    return details, impacts


def get_mitigation(mitigation_data: list[dict[str, str]], cwe_id: str) -> list[str]:
    """
    Extracts and returns a list of mitigation descriptions for a given CWE ID.

    Args:
        mitigation_data (list[dict[str, str]]): A list of dictionaries containing mitigation data.
        cwe_id (str): The CWE ID for which to extract mitigation descriptions.

    Returns:
        list[str]: A list of descriptions of potential mitigations for the given CWE ID.
    """
    # extract descriptions of potential mitigations only, ignoring phases
    potential_mitigations = "".join(
        [
            entry["potential_mitigations"]
            for entry in mitigation_data
            if entry["id"] == cwe_id
        ]
    )
    mitigations = re.findall(r"DESCRIPTION: (.+)", potential_mitigations)
    return mitigations


def process_pdf(pdf_path: str) -> list[str]:
    """
    Processes a PDF document to identify vulnerabilities, map them to business processes and CWEs, and suggest mitigations.
    This is the core of the engine that processes the PDF document and generates responses based on the identified vulnerabilities.

    Args:
        pdf_path (str): The path to the PDF document to be processed.

    Returns:
        list[str]: A list of responses generated during the processing, including identified vulnerabilities, mapped business processes, CWE rankings, and suggested mitigations.

    Raises:
        FileNotFoundError: If the vector databases are not found.
        RuntimeError: If there is an error parsing the response.
    """
    out: list[str] = []
    logger = logging.getLogger("gai")

    llm = ChatOpenAI(model="gpt-4o")
    llm_mini = ChatOpenAI(model="gpt-4o-mini")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    store_process, store_cwe = get_vectorstores(embeddings)
    if store_cwe is None or store_process is None:
        raise FileNotFoundError("Vector db not found. Please build the db first.")

    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    store = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
    retriever = store.as_retriever()

    system_prompt = """
    Answer the question based only on the context provided.
    Do not repeat yourself.

    Your response should match the following format:
    Number: <number>
    Vulnerability: <vulnerability>
    Explanation: <explanation>
    Impact: <impact on the organization>

    Context: {context}
    """

    question = """
    You are a threat intelligence analyst. Look at the breach report provided.
    what are the vulnerable business process or vulnerabilities identified that lead to a compromise? Be as specific as possible.
    You should state each vulnerability in a sentence, and provide a short description not exceeding 2 sentences.
    Include the impact of the vulnerability on the organization in your explanation.
    if the vulnerability stems from outdated software or a vulnerability in an external software, state that it is a dependency on a vulnerable third-party component and a violation of Secure Design Principles in your explanation.
    Do not repeat the vulnerabilities and do not provide any additional information.
    If the identified vulnerability is an in-depth version of a previously identified vulnerability, ignore it and move on.
    If the vulnerability is reasonably inferred from a previous vulnerability, ignore it and move on.
    Focus on unique vulnerabilities that have not been previously identified.
    You should list each vulnerability in a numerical order and follow the format provided strictly.
    """
    response = prompt_model(llm, system_prompt, retriever, question)
    logger.info("---vulnerabilities identified---")
    logger.info(response)
    logger.info("---model response end---\n\n")
    out.append(response)
    # parse previous response
    response_parsed, impacts = get_relevant_details(response)
    if response_parsed is None or impacts is None:
        raise RuntimeError(
            "Error parsing response. Please check the response format.", response
        )

    logger.debug("---vulnerabilities parsed---")
    for i in response_parsed:
        logger.debug(f"vuln identified: {i}")

    # Step 1: Map vulnerabilities to process categories
    top_n_processes = map_vulns_to_processes(response_parsed, store_process, 3)
    system_prompt_process = """
    Answer the question based only on the context provided.

    Your response should match the following format:
    Vulnerability Identified: <vulnerability>
    Process Category ID: <ID>
    Process Category Name: <Name>
    Explanation: <explanation>

    Context: {context}
    """
    for mapping in top_n_processes:
        logger.debug("---vectorstore search results---")
        logger.debug(f"vulnerability: {mapping.vulnerability}")
        logger.debug(f"process: {[x for x in mapping.process_ids]}")
        logger.debug(f"score: {[x for x in mapping.scores]}\n")
        logger.debug("---vectorstore search results end---")

        retriever = DocArrayInMemorySearch.from_documents(
            mapping.docs, embedding=embeddings
        ).as_retriever()

        question_process = f"""
        You are a threat intelligence analyst.
        You have previously identified the following vulnerability:
        {mapping.vulnerability}

        Based on the context provided, identify the business process most relevant to the cause of the vulnerability.
        Include the identified vulnerability in the response under "Vulnerability Identified".
        Format your response according to the prompt.
        """

        response_process = prompt_model(
            llm_mini, system_prompt_process, retriever, question_process
        )
        logger.info("---Business Process Mapping---")
        logger.info(response_process)
        logger.info("---model response end---\n")
        out.append(response_process)

    # Step 2: Map vulnerabilities to CWEs

    top_n_cwes = map_vulnerabilities_to_cwe(response_parsed, store_cwe, 6, impacts)
    system_prompt_cwe = """
    Answer the question based only on the context provided.

    Your response should match the following format only:
    Ranking: <ranking>
    Vulnerability Identified: <vulnerability>
    CWE ID: <CWE ID>
    CWE Description: <description>
    Explanation: <explanation>

    Context: {context}
    """
    cwe_responses: list[str] = []
    for mapping in top_n_cwes:
        logger.debug("---vectorstore search results---")
        logger.debug(f"vulnerability: {mapping.vulnerability}")
        logger.debug(f"cwe: {[x for x in mapping.cwe_ids]}")
        logger.debug(f"score: {[x for x in mapping.scores]}\n")
        logger.debug("---vectorstore search results end---")

        retriever = DocArrayInMemorySearch.from_documents(
            mapping.docs, embedding=embeddings
        ).as_retriever()

        question_cwe = f"""
        You are a threat intelligence analyst.
        You have previously identified the following vulnerability:
        {mapping.vulnerability}
        impact: {mapping.impact}

        Based on the context provided, rank the CWE IDs that best describes the vulnerability.
        You should only list the top 3 CWE IDs.
        Your decision should be informed by the technical details, keywords, and context provided.
        Include the identified vulnerability in the response under "Vulnerability Identified".
        If the CWEs provided are not relevant to the vulnerability, prioritize the CWEs that best explain the impact of the vulnerability on the organization.
        Format your response according to the format in the prompt.
        """

        response_cwe = prompt_model(llm, system_prompt_cwe, retriever, question_cwe)
        logger.info("---CWE ranking---")
        logger.info(response_cwe)
        logger.info("---model response end---\n")
        cwe_responses.append(response_cwe)
        out.append(response_cwe)

    mitigations = load_cwe_mitigations("cwe_dict_clean.json")
    system_prompt_mitigation = """
    Your response should match the following format:

    Vulnerability: <vulnerability>
    CWE ID: <CWE ID>
    Mitigation: ***<Mitigation>***
    Explanation: <explanation>

    context: {context}
    """
    for response in cwe_responses:
        vulnerabilities = re.findall(r"Vulnerability Identified: (.+)", response)
        cwe_id = re.findall(r"CWE ID: (.+)", response)
        if len(vulnerabilities) != len(cwe_id):
            raise RuntimeError(
                "Number of vulnerabilities and CWE IDs do not match. Check the response format.",
                response,
            )
        vuln_mitigations: list[str] = []
        for i in range(len(vulnerabilities)):
            mitigation = "".join(get_mitigation(mitigations, cwe_id[i].strip()))
            if mitigation == "":
                question_mitigation = f"""
                You are a threat intelligence analyst.
                You have previously identified the following vulnerability:
                Vulnerability Identified: {vulnerabilities[i]}
                CWE ID :{cwe_id[i]}

                Based on the context provided, you are to come up with the mitigation for the described vulnerability.
                Mitigations provided should tackle the root cause of the vulnerability.
                Format your response according to the prompt.
                """
                response_mitigation = prompt_model(
                    llm, system_prompt_mitigation, retriever, question_mitigation
                )
                vuln_mitigation = f"{response_mitigation}\n"
            else:
                vuln_mitigation = f"Vulnerability: {vulnerabilities[i]}\nCWE ID: {cwe_id[i]}\nMitigation: {mitigation}\n"
            vuln_mitigations.append(vuln_mitigation)
        logger.info("---Solutions---")
        logger.info("\n".join(vuln_mitigations))
        logger.info("---model response end---\n")
        out.append("\n".join(vuln_mitigations))
    return out


def eval_mode(pdf_paths: str):
    """
    Processes all PDF files in the given directory and writes the results to log files.

    Args:
        pdf_paths (str): The path to the directory containing PDF files.

    The function searches for all PDF files in the specified directory and its subdirectories.
    For each PDF file found, it processes the file twice and writes the results to separate log files.
    The log files are named based on the original PDF file name with a suffix `_01.log` and `_02.log`.
    """
    pdfs = Path(pdf_paths).rglob("*.pdf")

    for pdf in pdfs:
        for i in range(1, 3):
            out_path = Path(pdf_paths) / f"{pdf.name[:-4]}_0{i}.log"
            try:
                pdf_result = process_pdf(pdf.as_posix())
            except Exception as e:
                logger.error(f"Error processing {pdf.name}: {e}")
                continue
            with open(out_path, "w") as f:
                f.write("\n\n".join(pdf_result))


def main():
    # set up logging to log messages only from this module
    args = parse_args()
    logger = logging.getLogger("gai")
    logger.setLevel(args.log)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(sh)

    if "OPENAI_API_KEY" not in environ:
        raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")

    if args.eval:
        eval_mode(args.pdf_path)
        return
    process_pdf(args.pdf_path)


if __name__ == "__main__":
    main()
