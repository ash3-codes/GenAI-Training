from ingestion.loader import PDFLoader
from ingestion.structure_parser import StructureParser
from ingestion.chunker import SemanticChunker
from ingestion.metadata_builder import MetadataBuilder
from vectorstore.indexer import AzureQdrantIndexer
from config.settings import POLICY_DOCS_PATH

import ingestion.structure_parser
print(ingestion.structure_parser.__file__)

def run():

    loader = PDFLoader(POLICY_DOCS_PATH)
    parser = StructureParser()
    chunker = SemanticChunker()
    metadata_builder = MetadataBuilder()
    indexer = AzureQdrantIndexer()

    raw_docs = loader.load_pdfs()

    all_blocks = []

    for doc in raw_docs:
        for page in doc["pages"]:

            structured_blocks = parser.parse_page(page["text"])

            print("Page:", page["page_number"])
            print("Blocks returned:", len(structured_blocks))

            for block in structured_blocks:
                block.update({
                    "doc_name": doc["doc_name"],
                    "page_number": page["page_number"]
                })

            all_blocks.extend(structured_blocks)

    print("Total structured blocks:", len(all_blocks))

    chunked_blocks = chunker.chunk_all_blocks(all_blocks)
    print("Total chunked blocks:", len(chunked_blocks))

    enriched_docs = metadata_builder.enrich_all(chunked_blocks)
    print("Total enriched docs:", len(enriched_docs))

    indexer.index_documents(enriched_docs)


if __name__ == "__main__":
    run()