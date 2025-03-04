from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams,LLMTestCase
from langchain_community.document_loaders import PyPDFLoader
import json
import re

def get_pdf_contents(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    return [page.page_content for page in pages]

def evaluate_vuln_identification(pdf_contents : list, params: dict):
    vuln_identification_metric = GEval(
    name="Vulnerability_Identification",
    criteria="Determine if the 'actual output' has correctly identified the vulnerabilites based on the 'context' as compared to the 'expected output'. If the 'actual output' is more relevant and correct than the 'expected output' then score it higher",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.CONTEXT],
    threshold=0.5
)
    test_case = LLMTestCase(
    input=
    """ You are a threat intelligence analyst. Look at the breach report provided.
        what are the vulnerable business process or vulnerabilities identified that lead to a compromise? Be as specific as possible.
        You should state each vulnerability in a sentence, and provide a short description not exceeding 2 sentences.
        Include the impact of the vulnerability on the organization in your explanation.
        if the vulnerability stems from outdated software or a vulnerability in an external software, state that it is a dependency on a vulnerable third-party component and a violation of Secure Design Principles in your explanation.
        Do not repeat the vulnerabilities and do not provide any additional information.
        If the identified vulnerability is an in-depth version of a previously identified vulnerability, ignore it and move on.
        If the vulnerability is reasonably inferred from a previous vulnerability, ignore it and move on.
        Focus on unique vulnerabilities that have not been previously identified.
    """,
    actual_output=params['vuln_llm_ans'],
    expected_output=params['vuln_human_ans'],
    context=pdf_contents
)
    vuln_identification_metric.measure(test_case)
    return vuln_identification_metric.score, vuln_identification_metric.reason

def evaluate_cwe_mapping(params: dict):
    cwe_mapping_metric = GEval(
    name="CWE_Mapping",
    criteria="Determine the relevancy of the CWE-IDs mappings in 'actual output' given the vulnerabilities listed in 'context' as compared to the ones identified in 'expected output'. If the 'actual output' that are not present in 'expected output' is more relevant to the vulnerability in 'context' then don't penalize it",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.CONTEXT],
    threshold=0.5
    )
    test_case = LLMTestCase(
    input="""
        You are a threat intelligence analyst.
        You have previously identified the following vulnerability:

        Based on the context provided, choose the CWE IDs that best describes the vulnerability.
        Your decision should be informed by the technical details, keywords, and context provided.
        Include the identified vulnerability in the response under "Vulnerability Identified".
        If the CWEs provided are not relevant to the vulnerability, prioritize the CWEs that best explain the impact of the vulnerability on the organization.
        """,
    actual_output=params['cwe_llm_ans'],
    expected_output=params['cwe_human_ans'],
    context=params['vuln_human_ans'],
)
    cwe_mapping_metric.measure(test_case)
    return cwe_mapping_metric.score, cwe_mapping_metric.reason

def evaluate_business_process_mappings(params: dict):
    business_process_mappings_metric = GEval(
    name="Business_Process_Mapping",
    criteria="Determine the accuracy of the mapping of Business Process IDs in the 'actual output' as compared to the 'expected output'. Use the vulnerabilities identified in 'context' and if there are more contents in 'actual output' than 'expected output' just check if they match and ignore the extra ones",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.CONTEXT],
    threshold=0.5
    )
    test_case = LLMTestCase(
    input="""
        You are a threat intelligence analyst.
        You have previously identified the following vulnerabilities
        Based on the context provided, identify the business process most relevant to the cause of the vulnerability.
        """,
    actual_output=params['bp_llm_ans'],
    expected_output=params['bp_human_ans'],
    context=params['vuln_human_ans'],
)

    business_process_mappings_metric.measure(test_case)
    return business_process_mappings_metric.score, business_process_mappings_metric.reason

def processing(filelist):
    output = {}
    uncleaned = ["vuln_human_ans","bp_human_ans","cwe_human_ans"]
    cleaned = ["vuln_llm_ans","bp_llm_ans","cwe_llm_ans"]
    for file in filelist:
        with open("docs/" + file[:-4] + "_01.log", 'r', encoding='utf-8') as f:
            text = f.read()
            sections = re.split(r'```', text)
            cleaned_sections = [" ".join(section.split()) for section in sections if section.strip()]
        section_dict = {}
        for i in uncleaned:
            section_dict[i] = []
        for idx,i in enumerate(cleaned_sections):
            section_dict[cleaned[idx]] = i
    output[file] =section_dict
    with open("output_engine_1.json", "w") as f:
        json.dump(output, f, indent=4)

with open("output_base_1.json","r",encoding='utf8') as f:
    params = json.load(f)

with open("filelist.txt","r") as f:
    files = [i.strip() for i in f.readlines()]

for file in files:
    fields = params[file]
    print("Evaluating: ", file)
    pdf_contents = get_pdf_contents('docs/' + file)
    print(evaluate_vuln_identification(pdf_contents,fields))
    print(evaluate_business_process_mappings(fields))
    print(evaluate_cwe_mapping(fields))
    print("\n")


# Filelist.txt contains the file names
# All pdfs and logs are put in the doc folder - not decoupled properly yet
# The log file needs to be seperated with a ``` so like Vulns + ``` + BP + ``` + CWE
# the log files also must match the file name
# Processing will output a json file where it containts PDF File Name : {llm_ans : ans}
# human answer will be manually filled in as a list of strings
#for processing run with UV
#for solutions run with deepeval test run pyfilename