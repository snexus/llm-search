from llmsearch.config import SemanticSearchConfig, ObsidianAdvancedURI, OutputModel, SemanticSearchOutput


def get_and_parse_response(prompt: str, chain, embed_retriever, config: SemanticSearchConfig) -> OutputModel:
    most_relevant_docs = []
    docs = embed_retriever.get_relevant_documents(query=prompt)
    len_ = 0

    for doc in docs:
        doc_length = len(doc.page_content)
        if len_ + doc_length < config.max_char_size:
            most_relevant_docs.append(doc)
            len_ += doc_length
    res = chain({"input_documents": most_relevant_docs, "question": prompt}, return_only_outputs=False)

    out = OutputModel(response=res['output_text'])
    for doc in res["input_documents"]:
        doc_name = doc.metadata["source"]
        doc_name = doc_name.replace(
            config.replace_output_path.substring_search, config.replace_output_path.substring_replace
        )

        if config.obsidian_advanced_uri is not None:
            doc_name = process_obsidian_uri(doc_name, config.obsidian_advanced_uri, doc.metadata)
        
        text = doc.page_content
        out.semantic_search.append(SemanticSearchOutput(chunk_link=doc_name, chunk_text=text))
    return out

def process_obsidian_uri(doc_name: str, adv_uri_config: ObsidianAdvancedURI, metadata: dict) -> str:
    """Adds a suffix pointing to a specific heading based on the metadata supplied if doc.metadata

    Args:
        doc_name (str): Document name (partially processed, potentially)
        adv_uri_config (ObsidianAdvancedURI): contains the template to add, matches Obsidian's advanced URI plugin schem
        metadata (dict): Metadata associated with a document.

    Returns:
        str: document name with a header suffix.
    """
    print(metadata)
    append_str = adv_uri_config.append_heading_template.format(heading = metadata['heading'])
    return doc_name + append_str
    
    
