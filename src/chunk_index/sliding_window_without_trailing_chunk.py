import fitz  # PyMuPDF
import textwrap
import os
from typing import List

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: A single string containing all text from the PDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file at path '{pdf_path}' was not found.")

    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
        return ""

    return text

def sliding_window_chunker_words(text: str, chunk_size_words: int, overlap_words: int) -> List[str]:
    """
    Splits a given text into chunks using a sliding window approach based on word count.

    Args:
        text (str): The input text to be chunked.
        chunk_size_words (int): The maximum number of words for each chunk.
        overlap_words (int): The number of words to overlap between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    if overlap_words >= chunk_size_words:
        raise ValueError("Overlap must be less than chunk size.")
    
    words = text.split()
    chunks = []
    start_index = 0
    num_words = len(words)
    
    while start_index < num_words:
        end_index = min(start_index + chunk_size_words, num_words)
        chunk_words = words[start_index:end_index]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        
        # This conditional ensures that even a small final chunk is included
        # if end_index >= num_words:
        #     break
            
        start_index += (chunk_size_words - overlap_words)
        
    return chunks

# --- Example Usage ---
# IMPORTANT: Replace 'CCPA_Regulation_Document.pdf' with the actual path to your PDF file.
PDF_FILE_PATH = "/home/trupti/Downloads/Lexguard PE/CCPA_Regulation_Document.pdf"
CHUNK_SIZE_WORDS = 100  
OVERLAP_WORDS = 20      

print("Starting PDF text extraction...")
# Step 1: Extract all text from the PDF
full_text = extract_text_from_pdf(PDF_FILE_PATH)

if full_text:
    print("Text extraction successful. Starting word-based chunking...")
    # Step 2: Chunk the extracted text using the new word-based function
    chunks = sliding_window_chunker_words(full_text, CHUNK_SIZE_WORDS, OVERLAP_WORDS)
    
    print(f"\nOriginal Text Word Count: {len(full_text.split())}")
    print(f"Generated {len(chunks)} chunks with size {CHUNK_SIZE_WORDS} and overlap {OVERLAP_WORDS}:\n")
    
    # Print the first few chunks for a quick check
    for i, chunk in enumerate(chunks[:5]):
        print(f"--- Chunk {i+1} ({len(chunk.split())} words) ---")
        print(textwrap.fill(chunk, 80))
        print("-" * 20)
else:
    print("Could not extract text from the PDF. Please check the file path and format.")