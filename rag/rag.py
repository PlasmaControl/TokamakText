import os
from text_helpers import document_info
from llm_interface import LLMInterface, get_llm_interface
import chromadb

# Load environment variables at the start of your script to ensure all settings are correctly initialized.
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

# Reading prompts from files and storing them in constants.
with open('prompts/system_prompt.txt', 'r') as f:
    SYSTEM_PROMPT = f.read()
with open('prompts/user_prompt.txt', 'r') as f:
    USER_PROMPT = f.read()
with open('prompts/query_system_prompt.txt', 'r') as f:
    QUERY_SYSTEM_PROMPT = f.read()

# Initialize the database client.
client = chromadb.PersistentClient(path="db/")
print(f"{client.list_collections()}")

def retrieve(question: str):
    """
    Retrieves relevant documents based on a query from a chromaDB collection.

    Parameters
    ----------
    question : str
        The question or query to retrieve documents for.

    Returns
    -------
    dict
        A dictionary mapping document IDs to their corresponding text.
    """
    print(f'initial question: {question}')
    res = {}
    for document_type in ['shot', 'run', 'miniproposal']:
        n_results = document_info[document_type]['n_documents']
        if n_results > 0:
            collection = client.get_collection(f'{document_type}_embeddings')
            qr = collection.query(query_texts=question, n_results=n_results)
            ids = qr['ids'][0]
            documents = qr['documents'][0]
            res.update({k: v for k, v in zip(ids, documents)})
    return res

def process_results(results: dict) -> str:
    """
    Processes results into a formatted string for output or further processing.

    Parameters
    ----------
    results : dict
        A dictionary containing document IDs and their respective contents.

    Returns
    -------
    str
        A formatted string representation of the results.
    """
    processed_results = ""
    for k, v in results.items():
        processed_results += f"{k}: {v}\n"
    return processed_results

def rag_answer_question(question: str, results: dict, model: LLMInterface) -> str:
    """
    Generates an answer to a question using the retrieved documents and a language model interface.

    Parameters
    ----------
    question : str
        The user's question.
    results : dict
        The retrieved documents to base the answer on.
    model : LLMInterface
        The language model interface to generate the answer.

    Returns
    -------
    str
        The generated answer to the question.
    """
    processed_results = process_results(results)
    formatted_user_prompt = USER_PROMPT.format(question=question, results=processed_results)
    print(formatted_user_prompt)
    return model.query(SYSTEM_PROMPT, formatted_user_prompt)

def test():
    """
    Test function to demonstrate the retrieval and question answering process.
    """
    question = "Tell me about shots that struggled with tearing modes"
    model = get_llm_interface("openai")
    results = retrieve(question)
    answer = rag_answer_question(question, results, model)
    print(f"Model {model.model_name} answer:\n{answer}")

if __name__ == '__main__':
    test()