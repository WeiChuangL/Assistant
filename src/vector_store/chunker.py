import re
from dataclasses import dataclass

from src.config import settings


@dataclass
class Chunk:
    content: str
    index: int
    metadata: dict | None = None


def chunk_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[Chunk]:
    """Split text into overlapping chunks, trying to break at paragraph/sentence boundaries."""
    chunk_size = chunk_size or settings.agent_chunk_size
    chunk_overlap = chunk_overlap or settings.agent_chunk_overlap

    # Split into paragraphs first
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[Chunk] = []
    current = ""
    index = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) <= chunk_size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(Chunk(content=current, index=index))
                index += 1
            # If a single paragraph exceeds chunk_size, split by sentence
            if len(para) > chunk_size:
                sub_chunks = _split_long_paragraph(para, chunk_size, chunk_overlap)
                for sc in sub_chunks:
                    sc.index = index
                    chunks.append(sc)
                    index += 1
                current = ""
            else:
                current = para

    if current:
        chunks.append(Chunk(content=current, index=index))

    return chunks


def _split_long_paragraph(para: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Split a long paragraph into chunks by sentence boundaries."""
    sentences = re.split(r"(?<=[。！？.!?])\s*", para)
    chunks: list[Chunk] = []
    current = ""
    idx = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) <= chunk_size:
            current = (current + sent).strip() if current else sent
        else:
            if current:
                chunks.append(Chunk(content=current, index=idx))
                idx += 1
            current = sent

    if current:
        chunks.append(Chunk(content=current, index=idx))

    return chunks
