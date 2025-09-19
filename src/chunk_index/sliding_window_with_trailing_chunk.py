import fitz  # PyMuPDF
import re
import textwrap
from typing import List

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)

def normalize_text(text: str) -> str:
    """
    - Remove hyphenation at a line break (e.g. "infor-\n mation" -> "information")
    - Replace newlines with spaces and collapse multiple whitespace
    """
    text = re.sub(r'-\s*\n\s*', '', text)   # remove hyphen + newline
    text = re.sub(r'\n+', ' ', text)        # convert newlines to spaces
    text = re.sub(r'\s+', ' ', text).strip()# collapse whitespace
    return text

def sliding_window_chunker_words(
    text: str,
    chunk_size_words: int,
    overlap_words: int,
    min_last_chunk_size: int = 0  # optional: if last chunk < this, merge into previous
) -> List[str]:
    """
    Sliding-window chunker (word-count based).

    If min_last_chunk_size > 0 and the final chunk is smaller than that,
    the final chunk will be merged into the previous chunk.
    """
    if overlap_words >= chunk_size_words:
        raise ValueError("Overlap must be less than chunk size.")

    words = text.split()
    num_words = len(words)
    step = chunk_size_words - overlap_words
    chunks = []

    for start in range(0, num_words, step):
        end = start + chunk_size_words
        chunk_words = words[start:end]  # Python slicing clamps automatically
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))

    # Optional: merge tiny final chunk into previous chunk
    if min_last_chunk_size > 0 and len(chunks) > 1:
        last_wc = len(chunks[-1].split())
        if last_wc < min_last_chunk_size:
            chunks[-2] = chunks[-2] + " " + chunks[-1]
            chunks.pop()

    return chunks

# --- Example usage ---
PDF_FILE_PATH = "/home/trupti/Downloads/Lexguard PE/Indian Oil Priivacy policy.pdf"
CHUNK_SIZE_WORDS = 100
OVERLAP_WORDS = 20
MIN_LAST_CHUNK_SIZE = 10  # set to >0 if you want to avoid tiny trailing chunks

raw = extract_text_from_pdf(PDF_FILE_PATH)
normalized = normalize_text(raw)
chunks = sliding_window_chunker_words(normalized, CHUNK_SIZE_WORDS, OVERLAP_WORDS, MIN_LAST_CHUNK_SIZE)

print("Normalized Word Count:", len(normalized.split()))
print(f"Generated {len(chunks)} chunks.\n")

for i, chunk in enumerate(chunks, 1):
    print(f"--- Chunk {i} ({len(chunk.split())} words) ---")
    print(textwrap.fill(chunk, 80))
    print("-" * 40)