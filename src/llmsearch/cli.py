from llmsearch.vector_stores import VectorStoreChroma
import click

@click.group(name="index")
def index_group():
    """Index commands."""

@click.group
def main_cli():
    pass

@click.command(name="create")
@click.option(
    "--document-folder",
    "-d",
    "document_folder",
    required = True,
    type=click.Path(exists=True, dir_okay=True, writable=True),
    help="Specifies a document folder to scan.",
)
@click.option(
    "--output-embedding-folder",
    "-o",
    "output_embedding_folder",
    required = True,
    type=click.Path(exists=True, dir_okay=True, writable=True),
    help="Specifies a folder to store the embeddings.",
)
@click.option(
    "--embedding-model-name",
    "-m",
    "embed_model_name",
    default="all-MiniLM-L6-v2",
    help="Specifies a folder to store the embeddings."
)
@click.option(
    "--scan-extension",
    "-e",
    "scan_extension",
    default="md",
    help="Specifies a folder to store the embeddings."
)
def generate_index(document_folder, output_embedding_folder, 
                   embed_model_name: str = "all-MiniLM-L6-v2", scan_extension: str = "md"):
    vs = VectorStoreChroma(persist_folder=output_embedding_folder, hf_embed_model_name=embed_model_name)
    vs.create_index_from_folder(folder_path=document_folder, extension=scan_extension)
 
index_group.add_command(generate_index)

# add command groups to CLI root
main_cli.add_command(index_group)

if __name__ == '__main__':
    main_cli()