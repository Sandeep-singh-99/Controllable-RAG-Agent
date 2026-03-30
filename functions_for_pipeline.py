from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import OllamaLLM
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
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cuda"}, encode_kwargs = {'normalize_embeddings': False})
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

def create_keep_only_relevant_content_chain():
      keep_only_relevant_content_prompt_template = """you receive a query: {query} and retrieved docuemnts: {retrieved_documents} from a
       vector store.
       You need to filter out all the non relevant information that don't supply important information regarding the {query}.
       your goal is just to filter out the non relevant information.
       you can remove parts of sentences that are not relevant to the query or remove whole sentences that are not relevant to the query.
       DO NOT ADD ANY NEW INFORMATION THAT IS NOT IN THE RETRIEVED DOCUMENTS.
       output the filtered relevant content.
       """

      class KeepRelevantContent(BaseModel):
          relevant_content: str = Field(
              description="The relevant content from the retrieved documents that is relevant to the query.")

      keep_only_relevant_content_prompt = PromptTemplate(
              template=keep_only_relevant_content_prompt_template,
              input_variables=["query", "retrieved_documents"]
      )

      keep_only_relevant_content_llm = OllamaLLM(model="gemma3:12b", temperature=0, num_gpu=1)
      keep_only_relevant_content_chain = keep_only_relevant_content_prompt | keep_only_relevant_content_llm.with_structured_output(KeepRelevantContent)
      return keep_only_relevant_content_chain

keep_only_relevant_content_chain = create_keep_only_relevant_content_chain()

def keep_only_relevant_content(state):
    """
       Keeps only the relevant content from the retrieved documents that is relevant to the query.

       Args:
           question: The query question.
           context: The retrieved documents.
           chain: The LLMChain instance.

       Returns:
           The relevant content from the retrieved documents that is relevant to the query.
    """

    question = state['question']
    context = state['context']

    input_data = {
        "query": question,
        "retrieved_documents": context
    }

    print("Keeping only the relevant content...")
    print("----------------------------")
    output = keep_only_relevant_content_chain.invoke(input_data)
    relevant_content = output.relevant_content
    relevant_content = "".join(relevant_content)
    relevant_content = escape_quotes(relevant_content)

    return {"relevant_context": relevant_content, "context": context, "question": question}


def create_question_answer_from_context_cot_chain():
    class QuestionAnswerFromContext(BaseModel):
        answer_based_on_content: str = Field(description="generate an answer to a query based on a  given context.")

    question_answer_from_context_llm = OllamaLLM(model="gemma3:12b", temperature=0, num_gpu=1)

    question_answer_cot_prompt_template = """ 
       Examples of Chain-of-Thought Reasoning

       Example 1

       Context: Mary is taller than Jane. Jane is shorter than Tom. Tom is the same height as David.
       Question: Who is the tallest person?
       Reasoning Chain:
       The context tells us Mary is taller than Jane
       It also says Jane is shorter than Tom
       And Tom is the same height as David
       So the order from tallest to shortest is: Mary, Tom/David, Jane
       Therefore, Mary must be the tallest person

       Example 2
       Context: Harry was reading a book about magic spells. One spell allowed the caster to turn a person into an animal for a short time. Another spell could levitate objects.
       A third spell created a bright light at the end of the caster's wand.
       Question: Based on the context, if Harry cast these spells, what could he do?
       Reasoning Chain:
       The context describes three different magic spells
       The first spell allows turning a person into an animal temporarily
       The second spell can levitate or float objects
       The third spell creates a bright light
       If Harry cast these spells, he could turn someone into an animal for a while, make objects float, and create a bright light source
       So based on the context, if Harry cast these spells he could transform people, levitate things, and illuminate an area
       Instructions.

       Example 3 
       Context: Harry Potter woke up on his birthday to find a present at the end of his bed. He excitedly opened it to reveal a Nimbus 2000 broomstick.
       Question: Why did Harry receive a broomstick for his birthday?
       Reasoning Chain:
       The context states that Harry Potter woke up on his birthday and received a present - a Nimbus 2000 broomstick.
       However, the context does not provide any information about why he received that specific present or who gave it to him.
       There are no details about Harry's interests, hobbies, or the person who gifted him the broomstick.
       Without any additional context about Harry's background or the gift-giver's motivations, there is no way to determine the reason he received a broomstick as a birthday present.

       For the question below, provide your answer by first showing your step-by-step reasoning process, breaking down the problem into a chain of thought before arriving at the final answer,
       just like in the previous examples.
       Context
       {context}
       Question
       {question}
       """

    question_answer_from_context_cot_prompt = PromptTemplate(
        template=question_answer_cot_prompt_template,
        input_variables=["context", "question"]
    )

    question_answer_from_context_cot_chain = question_answer_from_context_cot_prompt | question_answer_from_context_llm.with_structured_output(QuestionAnswerFromContext)
    return question_answer_from_context_cot_chain

question_answer_from_context_cot_chain = create_question_answer_from_context_cot_chain()

def answer_question_from_context(state):
    """
        Answers a question from a given context.

        Args:
            question: The query question.
            context: The context to answer the question from.
            chain: The LLMChain instance.

        Returns:
            The answer to the question from the context.
    """

    question = state['question']
    context = state['aggregated_context'] if "aggregated_context" in state else state['context']

    input_data = {
        "context": context,
        "question": question
    }

    print("Answering the question from the retrieved context....")

    output = question_answer_from_context_cot_chain.invoke(input_data)
    answer = output.answer_based_on_content
    print(f"Answer before checking hallucination: {answer}")
    return {"answer": answer, "context": context, "question": question}

def create_is_relevant_content_chain():
    is_relevant_content_prompt_template = """you receive a query: {query} and a context: {context} retrieved from a vector store. 
        You need to determine if the document is relevant to the query. """

    class Relevance(BaseModel):
        is_relevant: bool = Field(description="Whether the document is relevant to the query.")
        explanation: str = Field(description="An explanation of why the document is relevant or not.")


    is_relevant_llm = OllamaLLM(model="gemma3:12b", temperature=0, num_gpu=1)

    is_relevant_content_prompt = PromptTemplate(
        template=is_relevant_content_prompt_template,
        input_variables=["query", "context"]
    )

    is_relevant_content_chain = is_relevant_content_prompt | is_relevant_llm.with_structured_output(Relevance)
    return is_relevant_content_chain

is_relevant_content_chain = create_is_relevant_content_chain()


def is_relevant_content(state):
    """
       Determines if the document is relevant to the query.

       Args:
           question: The query question.
           context: The context to determine relevance.
    """

    question = state['question']
    context = state['context']

    input_data = {
        "query": question,
        "context": context
    }

    # Invoke the chain to determine if the document is relevant
    output = is_relevant_content_chain.invoke(input_data)
    print("Determining if the document is relevant...")
    if output["is_relevant"] == True:
        print("The document is relevant.")
        return "relevant"
    else:
        print("The document is not relevant.")
        return "not relevant"

    # 280