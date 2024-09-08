import fitz
from unstructured.partition.pdf import partition_pdf

if __name__ == "__main__":
    n_output_elements = 2000
    path = "/storage/llm/pdf_docs/Patrick Viafore - Robust Python_ Write Clean and Maintainable Code-O'Reilly Media (2021).pdf"
    doc = fitz.open(path)
    with open("output_mupdf.txt", mode="w") as f:
        for i, page in enumerate(doc):
            block = page.get_text("block")
            f.write(block)
            f.write("**********\n")
        # if i == 100:
        #     break
        # print("-----------")
        # print(block)
        # print("-----------")
        # print(page.get_text("block"))

    elements = partition_pdf(
        filename=path,
        strategy="fast",
        include_page_breaks=True,
        infer_table_structure=True,
    )
    print(elements)
    print("Done processing. Writing output...")

    with open("output.txt", mode="w") as f:
        for el in elements:  # [:n_output_elements]:
            # s = f"<{el.category} - {el.metadata.page_number}> {el.text}\n"
            s = f"{el.text}\n"
            f.write(s)
