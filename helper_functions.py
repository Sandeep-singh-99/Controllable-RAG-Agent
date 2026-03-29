import tiktoken
import re
from langchain_community.docstore.document import Document
import PyPDF2
import pandas as pd
import textwrap

def num_tokens_from_string(string: str, encoding_name: str) -> str:
     """
    Calculates the number of tokens in a given string using a specified encoding.

    Args:
        string: The input string to tokenize.
        encoding_name: The name of the encoding to use (e.g., 'cl100k_base').

    Returns:
        The number of tokens in the string according to the specified encoding.
    """

     