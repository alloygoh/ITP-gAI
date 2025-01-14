from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sys import argv

BASE_URL = "http://10.147.18.3:11434"

def format_docs(docs):
  return "\n\n".join(doc.page_content for doc in docs)


llm = OllamaLLM(model='llama3.2:1B', base_url=BASE_URL)
embeddings = OllamaEmbeddings(model='bge-large', base_url=BASE_URL)

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

while True:
  # question = input('What do you want to learn from the document?\n')
  question = "You are a threat intelligence analyst. look only at the \"facts of this case\" section. what is the vulnerable business process or vulnerabilities identified that lead to a compromise? Be as specific as possible."
  print()
  print(chain.invoke(question))
  break
