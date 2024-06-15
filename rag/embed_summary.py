import sys
import h5py
from tqdm import tqdm
import chromadb
from text_helpers import make_document_dic_from_string

def read_file(file_name):
    """
    Opens and reads data from an HDF5 file.

    Parameters
    ----------
    file_name : str
        The file path to the HDF5 file.

    Returns
    -------
    h5py.File
        The opened HDF5 file object.
    """
    data = h5py.File(file_name, 'r')
    return data

def process_text_data(text_data):
    """
    Processes text data from an HDF5 dataset, reformatting and encoding it into a structured dictionary.

    Parameters
    ----------
    text_data : dict
        A dictionary where keys are runs and values are dictionaries containing textual data
        and associated metadata.

    Returns
    -------
    dict
        A dictionary with processed text where each run's data is formatted as a single string
        and converted to a document dictionary.
    """
    new_data = {}
    for run in text_data:
        strs = []
        brief = text_data[run]['brief'].decode('utf-8')
        for entry_ind in range(len(text_data[run]['text'])):
            text = text_data[run]['text'][entry_ind].decode('utf-8')
            topic = text_data[run]['topic'][entry_ind].decode('utf-8')
            username = text_data[run]['username'][entry_ind].decode('utf-8')
            strs.append(f"{brief}: user {username} ({topic}): {text}")
        new_data.update(make_document_dic_from_string("\n".join(strs), run, 'run'))
    return new_data

def main(target):
    """
    Main function to process HDF5 text data into a document database.

    Parameters
    ----------
    target : str
        The path to the target HDF5 file.

    Returns
    -------
    None
    """
    data = read_file(target)
    text_data = {}
    for run in tqdm(data):
        vals = data[run]
        text_data[run] = {
            'brief': vals['brief'][()],
            'text': vals['text'][:],
            'topic': vals['topic'][:],
            'username': vals['username'][:]
        }

    data.close()
    processed_text_data = process_text_data(text_data)
    client = chromadb.PersistentClient(path='db')
    collection = client.get_or_create_collection('run_embeddings')
    keys = list(processed_text_data.keys())
    values = list(processed_text_data.values())
    collection.add(ids=keys, documents=values)

if __name__ == '__main__':
    main(sys.argv[1])