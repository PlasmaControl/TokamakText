import sys
import os
import chromadb
from text_helpers import make_document_dic_from_string

def read_file(file_name):
    """
    Reads the entire content of a file.

    Parameters
    ----------
    file_name : str
        The path to the file that needs to be read.

    Returns
    -------
    str
        The content of the file as a string.
    """
    with open(file_name) as f:
        text = f.read()
    return text

def main(text_dir, document_type):
    """
    Processes all text files in a specified directory, creates document dictionaries, and stores them in a database.

    Parameters
    ----------
    text_dir : str
        The directory path that contains the text files.
    document_type : str
        The type of documents to be processed, used as a part of the collection name.

    Returns
    -------
    None
    """
    files = os.listdir(text_dir)
    text_data = {}
    for fn in files:
        full_path = os.path.join(text_dir, fn)
        text = read_file(full_path)
        text_data.update(make_document_dic_from_string(text, fn, document_type))

    client = chromadb.PersistentClient(path='db')
    collection = client.get_or_create_collection(f'{document_type}_embeddings')
    keys = list(text_data.keys())
    values = list(text_data.values())
    collection.add(ids=keys, documents=values)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])