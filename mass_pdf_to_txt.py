import PyPDF2
import os
from tqdm import tqdm
import shutil
import sys

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a PDF file.

    Parameters
    ----------
    pdf_path : str
        The file path of the PDF from which to extract text.

    Returns
    -------
    str
        All text extracted from the PDF.
    """
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:  # Only add text if page_text is not None
                text += page_text
    return text

def pdf_to_txt(input_pdf_path, output_txt_path):
    """
    Converts a PDF file to a text file by extracting all text and saving it.

    Parameters
    ----------
    input_pdf_path : str
        The file path of the input PDF.
    output_txt_path : str
        The file path for the output text file.
    """
    text_content = extract_text_from_pdf(input_pdf_path)
    with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text_content)

def convert_pdfs_to_text(pdf_dir, txt_dir):
    """
    Converts all PDF files in a directory to text files in another directory.

    Parameters
    ----------
    pdf_dir : str
        The directory that contains PDF files.
    txt_dir : str
        The directory where text files should be saved.
    """
    pdf_files = os.listdir(pdf_dir)
    for pdf in tqdm(pdf_files, desc="Converting PDFs to Text Files"):
        if pdf.lower().endswith('.pdf'):
            input_pdf_path = os.path.join(pdf_dir, pdf)
            output_txt_path = os.path.join(txt_dir, pdf[:-4] + '.txt')
            pdf_to_txt(input_pdf_path, output_txt_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <pdf_directory> <text_directory>")
        sys.exit(1)

    pdf_dir = sys.argv[1]
    txt_dir = sys.argv[2]
    
    # Ensure the output directory exists
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    
    convert_pdfs_to_text(pdf_dir, txt_dir)
    print("Files converted and saved successfully!")