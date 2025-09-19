import fitz  # PyMuPDF
import re
import textwrap
from typing import List
import nltk

# --- Run once to download sentence tokenizer ---
# nltk.download("punkt")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from all pages of a PDF."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)

def normalize_text(text: str) -> str:
    """
    - Fix hyphenation at line breaks ("infor-\nmation" -> "information")
    - Replace newlines with spaces
    - Collapse multiple spaces
    """
    text = re.sub(r'-\s*\n\s*', '', text)   # remove hyphen + newline
    text = re.sub(r'\n+', ' ', text)        # convert newlines to spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def sliding_window_sentence_chunks(
    text: str,
    chunk_size_words: int,
    overlap_words: int
) -> List[str]:
    """
    Splits text into chunks at sentence boundaries, using a sliding window.

    Args:
        text (str): input text
        chunk_size_words (int): approx max words per chunk
        overlap_words (int): number of words overlapped between chunks

    Returns:
        List[str]: sentence-aware chunks
    """
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_count = 0

    for sentence in sentences:
        sent_words = sentence.split()
        if current_count + len(sent_words) > chunk_size_words and current_chunk:
            # Save current chunk
            chunks.append(" ".join(current_chunk))

            # Keep overlap (last N words) and start new chunk
            overlap = current_chunk[-overlap_words:] if overlap_words > 0 else []
            current_chunk = overlap + sent_words
            current_count = len(current_chunk)
        else:
            current_chunk.extend(sent_words)
            current_count += len(sent_words)

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# --- Example usage ---
PDF_FILE_PATH = "/home/trupti/Downloads/Lexguard PE/CCPA_Regulation_Document.pdf"
CHUNK_SIZE_WORDS = 100
OVERLAP_WORDS = 20

raw_text = extract_text_from_pdf(PDF_FILE_PATH)
normalized_text = normalize_text(raw_text)

chunks = sliding_window_sentence_chunks(normalized_text, CHUNK_SIZE_WORDS, OVERLAP_WORDS)

print(f"Normalized word count: {len(normalized_text.split())}")
print(f"Generated {len(chunks)} chunks\n")

for i, chunk in enumerate(chunks, 1):  # print first 5 chunks
    print(f"--- Chunk {i} ({len(chunk.split())} words) ---")
    print(textwrap.fill(chunk, 80))
    print("-" * 40)