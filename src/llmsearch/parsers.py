import re
from pathlib import Path

from loguru import logger


def markdown_parser(file_path: Path, max_size = 2048):
    
    logger.info(f"Parsing {file_path}")
    
    with open(file_path, "r") as f:
        text = f.read()
                        
        sections = ['']
        
        # Split by code and non-code
        chunks = text.split("```")
        
        for i, chunk in enumerate(chunks):
            if i % 2 == 0: # Every even element is non-code
  
                headings = re.split(r'\n#\s+', chunk)              
                sections[-1]+=headings[0] # first split belongs to previous section
                
                if len(headings) > 1: # All next splits belong to subsequent sections
                    sections += headings[1:]
            else: # Process code section
                rows = chunk.split('\n')
                code = rows[1:]
                
                lang = rows[0] # Get the language name
                
                all_code = [f"Following is a code section in {lang}, delimited by triple backticks:", "```"] + code + ['```']
                sections[-1]+=("\n".join(all_code))
    
    logger.info(f"\tExtracted {len(sections)} sections.")
    
    sections_trimmed = []
    for section in sections:
        if len(section) > max_size:
            trimmed_chunks = split_by_size(section, max_size)
            sections_trimmed+=trimmed_chunks
        else:
            sections_trimmed.append(section)
        
    return sections_trimmed


def split_by_size(s, length: int):
    chunks = [s[i:i+length] for i in range(0, len(s), length)]
    return chunks
        

if __name__ == "__main__":
    path = Path("/shared/sample_data/markdown/apache-spark-programming-dataframes.md")
    d1 = markdown_parser(file_path=path)
    print(d1)
    