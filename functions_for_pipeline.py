from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END

from dotenv import load_dotenv
from pprint import pprint
import os
from typing_extensions import TypedDict
from typing import List, TypedDict

from helper_functions import escape_quotes, text_wrap

"""
Set the environment variables for the API Keys.
"""

load_dotenv()
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "100000"
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

def create_retrievers():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    chunks_vector_store = FAISS.load_local("chunks_vector_store", embeddings, allow_dangerous_deserialization=True)
    chapter_summaries_vector_store = FAISS.load_local("chapter_summarize_vector_store", embeddings, allow_dangerous_deserialization=True)
    book_quotes_vectorstore = FAISS.load_local("book_quotes_vectorstore", embeddings, allow_dangerous_deserialization=True)

    chunks_query_retriever = chunks_vector_store.as_retriever(search_kwargs={"k": 1})
    chapter_summaries_query_retriever = chapter_summaries_vector_store.as_retriever(search_kwargs={"k": 1})
    book_quotes_query_retriever = book_quotes_vectorstore.as_retriever(search_kwargs={"k": 10})
    return chunks_query_retriever, chapter_summaries_query_retriever, book_quotes_query_retriever

chunks_query_retriever, chapter_summaries_query_retriever, book_quotes_query_retriever = create_retrievers()


def retrieve_context_per_question(state):
    """
        Retrieves relevant context for a given question. The context is retrieved from the book chunks and chapter summaries.

        Args:
            state: A dictionary containing the question to answer.
    """

    print("Retrieving relevant chunks....")
    question = state["question"]
    docs = chunks_query_retriever.get_relevant_documents(question)

    context = " ".join(doc.page_content for doc in docs)

    print("Retrieving relevant chapter summaries....")
    docs_summaries = chapter_summaries_query_retriever.get_relevant_documents(state["question"])

    context_summaries = " ".join(f"{doc.page_content} (Chapter {doc.metadata['chapter']})" for doc in docs_summaries)

    print("Retrieving Relevant book quotes...")
    docs_book_quotes = book_quotes_query_retriever.get_relevant_documents(state["question"])
    book_quotes = " ".join(doc.page_content for doc in docs_book_quotes)

    all_contexts = context + context_summaries + book_quotes
    all_contexts = escape_quotes(all_contexts)

    return {"context": all_contexts, "question": question}



