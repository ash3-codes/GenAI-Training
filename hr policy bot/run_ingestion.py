# scripts/run_ingestion.py

from ingestion.loader import PDFLoader
from ingestion.structure_parser import StructureParser
from ingestion.chunker import SemanticChunker
from ingestion.metadata_builder import MetadataBuilder
from vectorstore.indexer import AzureQdrantIndexer

from config.settings import POLICY_DOCS_PATH


# ---------------------------------------------------------
# Early Filtering (Remove Low-Value Blocks BEFORE Chunking)
# ---------------------------------------------------------
def is_low_value_block(block):

    content = block.get("content", "")
    text = content.lower().strip()

    if not text:
        return True

    # Very short noise
    if len(text) < 40:
        return True

    # Table of contents
    if "table of contents" in text:
        return True

    if text.startswith("contents"):
        return True

    # Legal disclaimers
    if "do not disclose" in text:
        return True

    if "proprietary to" in text:
        return True

    # Cover page noise
    if "version" in text and len(text) < 200:
        return True

    return False


# ---------------------------------------------------------
# Main Ingestion Pipeline
# ---------------------------------------------------------
def run():

    loader = PDFLoader(POLICY_DOCS_PATH)
    parser = StructureParser()
    chunker = SemanticChunker()
    metadata_builder = MetadataBuilder()
    indexer = AzureQdrantIndexer()

    raw_docs = loader.load_pdfs()

    all_blocks = []

    # -------------------------------
    # Structure Parsing + Early Filter
    # -------------------------------
    for doc in raw_docs:
        for page in doc["pages"]:

            structured_blocks = parser.parse_page(page["text"])

            # Attach metadata
            for block in structured_blocks:
                block.update({
                    "doc_name": doc["doc_name"],
                    "page_number": page["page_number"]
                })

            # Early filtering
            filtered_blocks = [
                block for block in structured_blocks
                if not is_low_value_block(block)
            ]

            all_blocks.extend(filtered_blocks)

    print("Total structured blocks:", len(all_blocks))

    # -------------------------------
    # Chunking
    # -------------------------------
    chunked_blocks = chunker.chunk_all_blocks(all_blocks)
    print("Total chunked blocks:", len(chunked_blocks))

    # -------------------------------
    # Metadata Enrichment
    # -------------------------------
    enriched_docs = metadata_builder.enrich_all(chunked_blocks)
    print("Total enriched docs:", len(enriched_docs))

    # -------------------------------
    # Indexing
    # -------------------------------
    indexer.index_documents(enriched_docs)


if __name__ == "__main__":
    run()