import sys
import h5py
from tqdm import tqdm
import chromadb
from text_helpers import make_document_dic_from_string
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def read_file(file_name):
    """
    Opens and reads data from an HDF5 file.

    Parameters
    ----------
    file_name : str
        Path to the HDF5 file.

    Returns
    -------
    h5py.File
        The opened HDF5 file object.
    """
    data = h5py.File(file_name, 'r')
    return data

def process_text_data(text_data):
    """
    Processes textual data, extracting relevant information and converting it into a dictionary format.

    Parameters
    ----------
    text_data : dict
        Dictionary containing the text data with structure {shot: {'run', 'text', 'topic', 'username'}}.

    Returns
    -------
    dict
        New dictionary with processed text data.
    """
    new_data = {}
    for shot in text_data:
        strs = []
        run = text_data[shot]['run'].decode('utf-8')
        for entry_ind in range(len(text_data[shot]['text'])):
            text = text_data[shot]['text'][entry_ind].decode('utf-8')
            topic = text_data[shot]['topic'][entry_ind].decode('utf-8')
            username = text_data[shot]['username'][entry_ind].decode('utf-8')
            strs.append(f"user {username} ({topic}): {text}")
        new_data.update(make_document_dic_from_string("\n".join(strs), shot, 'shot'))
    return new_data

def main(target):
    """
    Main function to read, process, and store text data.

    Parameters
    ----------
    target : str
        Path to the HDF5 file to be processed.

    Returns
    -------
    None
    """
    data = read_file(target)
    text_data = {}
    for shot in tqdm(data):
        if shot in ('spatial_coordinates', 'times'):
            continue
        try:
            vals = data[shot]
            text_data[shot] = {
                'run': vals['run_sql'][()],
                'text': vals['text_sql'][:],
                'topic': vals['topic_sql'][:],
                'username': vals['username_sql'][:]
            }
        except Exception as e:
            print(f'{shot} failed: {str(e)}')

    data.close()
    processed_text_data = process_text_data(text_data)
    client = chromadb.PersistentClient(path="db")
    collection = client.get_or_create_collection("shot_embeddings")
    keys = list(processed_text_data.keys())
    values = list(processed_text_data.values())
    collection.add(ids=keys, documents=values)

if __name__ == '__main__':
    main(sys.argv[1])