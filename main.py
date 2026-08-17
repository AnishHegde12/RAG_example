import os

from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
import rich
load_dotenv()
api_key = os.getenv("API_KEY")

embeddings = CohereEmbeddings(
    model="embed-v4.0",
    cohere_api_key=api_key
)

loader=PyPDFLoader(r"TechCorp_Official_Employee_Handbook.pdf")
document=loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100, separators=["", " ", ".", "\n\n", "\n"])
text=text_splitter.split_documents(document)
db=Chroma.from_documents(text, embeddings)
retriever=db.as_retriever(search_type="similarity", search_kwargs={"k":3})
llm = ChatCohere(
    model="command-a-03-2025",
    cohere_api_key=api_key,
    temperature=0.7
)
chat_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Context:
{context}

Chat history:
{chat_history}
"""
    ),
    (
        "human",
        "{input}"
    )
])

doc_chain=create_stuff_documents_chain(llm,chat_prompt )
retriever_chain=create_retrieval_chain(retriever, doc_chain)
store={}
def get_session_history(session_id:str):
    if session_id not in store:
        store[session_id]=InMemoryChatMessageHistory()
    return store[session_id]
chain_with_memory=RunnableWithMessageHistory(
    retriever_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)
session_id="t1"
queries=[
"What days can I work from home?",
    "What are the work hours?",
    "Can we work remotely?",
    "Is there any L&D budget?",
    "What were my previous queries?"
]

for i, q in enumerate(queries,1):
    response=chain_with_memory.invoke({"input":q}, config={"configurable":{"session_id":session_id}})
    rich.print(f"\nQuery{i}:{q}")
    rich.print("Answer:",response['answer'])