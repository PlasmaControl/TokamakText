import re
import tiktoken
import os
from dotenv import load_dotenv

load_dotenv()

model = os.getenv("LLM_NAME")

# Define document information including token length and number of documents per type.
document_info = {
    'shot': {'token_length': 300, 'n_documents': 14},
    'run': {'token_length': 400, 'n_documents': 10},
    'miniproposal': {'token_length': 400, 'n_documents': 10}
}

overlap_num_tokens = 60  # Overlap tokens to prevent context loss in document splits.

def make_document_dic_from_string(input_string, in_key, document_type):
    """
    Processes and splits a string into smaller chunks based on token length for a specified document type.

    Parameters
    ----------
    input_string : str
        The input string to process and split.
    in_key : str
        A key identifying the document.
    document_type : str
        The type of document (e.g., 'shot', 'run', 'miniproposal').

    Returns
    -------
    dict
        A dictionary with keys as document identifiers and values as document contents.
    """
    # Normalize punctuation and repeated characters in the string.
    formatted_string = re.sub(r'([^a-zA-Z0-9\s]{2})[^a-zA-Z0-9\s]+', r'\1', input_string)
    formatted_string = re.sub(r',', ', ', formatted_string)
    words = re.split(r'\s+', formatted_string.strip())

    # Calculate the number of tokens allowed per document for the specified type.
    num_tokens = document_info[document_type]['token_length']
    encoder = tiktoken.encoding_for_model(model)

    # Encode the words, then decode them to handle potentially malformed byte sequences.
    encoded_documents = encoder.encode(' '.join(words))
    document_tokens = [
        encoder.decode_single_token_bytes(token).decode('utf-8', errors='ignore') for token in encoded_documents
    ]

    # Split the tokens into documents based on token count and overlap.
    array_of_documents = [
        ''.join(document_tokens[i:i + num_tokens])
        for i in range(0, len(document_tokens) - num_tokens + 1, num_tokens - overlap_num_tokens)
    ]

    # Create a dictionary of documents with unique keys.
    if len(array_of_documents) == 1:
        return {f'{document_type} {in_key}': array_of_documents[0]}
    else:
        return {f'{document_type} {in_key}_{i}': doc for i, doc in enumerate(array_of_documents)}