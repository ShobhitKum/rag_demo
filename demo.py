from langchain.text_splitter import Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


file_path = './temp/AWS.txt'
loader = TextLoader(file_path)
document = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=60,separators = ['\n'])
docs = text_splitter.split_documents(document)


embeddings = OllamaEmbeddings()
#db3 = Chroma.from_documents(documents=docs, embedding=embeddings,persist_directory="./chroma_db")
db3 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

retriever = db3.as_retriever(search_type="mmr",  # Also test "similarity"
     search_kwargs={"k": 8})

# retriever = embeddings.as_retriever(
#      search_type="mmr",  # Also test "similarity"
#      search_kwargs={"k": 8},
# )

#result = retriever.get_relevant_documents("Explain IAM policy ")

#######QA###########
from langchain import hub
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")

llm = Ollama(model="mistral",
             verbose=True,
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db3.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)
question = "What is Elastic fabric adapter?"
result2 = qa_chain({"query": question})


