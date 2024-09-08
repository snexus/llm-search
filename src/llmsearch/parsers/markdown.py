import re
import urllib
from collections import namedtuple
from enum import Enum
from pathlib import Path
from typing import Generator, List, Tuple, Union

from loguru import logger

FORMATTING_SEQUENCES = {"*", "**", "***", "_", "__", "~~", "||"}
CODE_BLOCK_SEQUENCES = {"`", "``", "```"}
ALL_SEQUENCES = FORMATTING_SEQUENCES | CODE_BLOCK_SEQUENCES
MAX_FORMATTING_SEQUENCE_LENGTH = max(len(seq) for seq in ALL_SEQUENCES)


class SplitCandidates(Enum):
    SPACE = 1
    NEWLINE = 2
    LAST_CHAR = 3


# Order of preference for splitting
SPLIT_CANDIDATES_PREFRENCE = [
    SplitCandidates.NEWLINE,
    SplitCandidates.SPACE,
    SplitCandidates.LAST_CHAR,
]


BLOCK_SPLIT_CANDIDATES = [r"\n#\s+", r"\n##\s+", r"\n###\s+"]
CODE_BLOCK_LEVEL = 10  # SHould be high enough to rank it above everything else

MarkdownChunk = namedtuple("MarkdownChunk", "string level")


class SplitCandidateInfo:
    last_seen: int
    active_sequences: List[str]
    active_sequences_length: int

    def __init__(self):
        self.last_seen = None
        self.active_sequences = []
        self.active_sequences_length = 0

    def process_sequence(self, seq: str, is_in_code_block: bool):
        """Process `seq`, update `self.active_sequences` and `self.active_sequences_length`,
        and return whether we are in a code block after processing `seq`.
        """

        if is_in_code_block:
            if self.active_sequences and seq == self.active_sequences[-1]:
                last_seq = self.active_sequences.pop()
                self.active_sequences_length -= len(last_seq)
            return True
        elif seq in CODE_BLOCK_SEQUENCES:
            self.active_sequences.append(seq)
            self.active_sequences_length += len(seq)
            return True
        else:
            for k in range(len(self.active_sequences) - 1, -1, -1):
                if seq == self.active_sequences[k]:
                    sequences_being_removed = self.active_sequences[k:]
                    self.active_sequences = self.active_sequences[:k]
                    self.active_sequences_length -= sum(
                        len(seq) for seq in sequences_being_removed
                    )
                    return False
            self.active_sequences.append(seq)
            self.active_sequences_length += len(seq)
            return False

    def copy_from(self, other):
        self.last_seen = other.last_seen
        self.active_sequences = other.active_sequences.copy()
        self.active_sequences_length = other.active_sequences_length


def phsyical_split(markdown: str, max_chunk_size: int) -> Generator[str, None, None]:
    """Naive markdown splitter that splits long messages in chunks
    preserving the markdown formatting tags. This split method isn't aware of higher level logical blocks,
    but preserves the low level blocks
    """

    if max_chunk_size <= MAX_FORMATTING_SEQUENCE_LENGTH:
        raise ValueError(
            f"max_chunk_size must be greater than {MAX_FORMATTING_SEQUENCE_LENGTH}"
        )

    split_candidates = {
        SplitCandidates.SPACE: SplitCandidateInfo(),
        SplitCandidates.NEWLINE: SplitCandidateInfo(),
        SplitCandidates.LAST_CHAR: SplitCandidateInfo(),
    }
    is_in_code_block = False

    chunk_start_from, chunk_char_count, chunk_prefix = 0, 0, ""

    def split_chunk():
        for split_variant in SPLIT_CANDIDATES_PREFRENCE:
            split_candidate = split_candidates[split_variant]
            if split_candidate.last_seen is None:
                continue
            chunk_end = split_candidate.last_seen + (
                1 if split_variant == SplitCandidates.LAST_CHAR else 0
            )
            chunk = (
                chunk_prefix
                + markdown[chunk_start_from:chunk_end]
                + "".join(reversed(split_candidate.active_sequences))
            )

            next_chunk_prefix = "".join(split_candidate.active_sequences)
            next_chunk_char_count = len(next_chunk_prefix)
            next_chunk_start_from = chunk_end + (
                0 if split_variant == SplitCandidates.LAST_CHAR else 1
            )

            split_candidates[SplitCandidates.NEWLINE] = SplitCandidateInfo()
            split_candidates[SplitCandidates.SPACE] = SplitCandidateInfo()
            return (
                chunk,
                next_chunk_start_from,
                next_chunk_char_count,
                next_chunk_prefix,
            )

    i = 0
    while i < len(markdown):
        for j in range(MAX_FORMATTING_SEQUENCE_LENGTH, 0, -1):
            seq = markdown[i : i + j]
            if seq in ALL_SEQUENCES:
                last_char_split_candidate_len = (
                    chunk_char_count
                    + split_candidates[
                        SplitCandidates.LAST_CHAR
                    ].active_sequences_length
                    + len(seq)
                )
                if last_char_split_candidate_len >= max_chunk_size:
                    (
                        next_chunk,
                        chunk_start_from,
                        chunk_char_count,
                        chunk_prefix,
                    ) = split_chunk()
                    yield next_chunk
                is_in_code_block = split_candidates[
                    SplitCandidates.LAST_CHAR
                ].process_sequence(seq, is_in_code_block)
                i += len(seq)
                chunk_char_count += len(seq)
                split_candidates[SplitCandidates.LAST_CHAR].last_seen = i - 1
                break

        if i >= len(markdown):
            break

        split_candidates[SplitCandidates.LAST_CHAR].last_seen = i
        chunk_char_count += 1
        if markdown[i] == "\n":
            split_candidates[SplitCandidates.NEWLINE].copy_from(
                split_candidates[SplitCandidates.LAST_CHAR]
            )
        elif markdown[i] == " ":
            split_candidates[SplitCandidates.SPACE].copy_from(
                split_candidates[SplitCandidates.LAST_CHAR]
            )

        last_char_split_candidate_len = (
            chunk_char_count
            + split_candidates[SplitCandidates.LAST_CHAR].active_sequences_length
        )
        if last_char_split_candidate_len == max_chunk_size:
            next_chunk, chunk_start_from, chunk_char_count, chunk_prefix = split_chunk()
            yield next_chunk

        i += 1

    if chunk_start_from < len(markdown):
        yield chunk_prefix + markdown[chunk_start_from:]


def get_logical_blocks_recursively(
    markdown: str, max_chunk_size: int, all_sections: list, split_candidate_index=0
) -> List[MarkdownChunk]:
    """Recursively scans blocks, splittling the larger blocks using next available paragraph size

    Args:
        markdown (str): Markdown to split
        max_chunk_size (int): Maximum chunk size
        all_sections (list): Keeps tracks of all sections
        split_candidate_index (int, optional): Index of split sequence in BLOCK_SPLIT_CANDIDATEs. Starts from 0.

    Returns:
        List[str]: List of logically split chunks
    """

    if split_candidate_index >= len(BLOCK_SPLIT_CANDIDATES):
        for chunk in phsyical_split(markdown, max_chunk_size):
            all_sections.append(
                MarkdownChunk(string=chunk, level=split_candidate_index)
            )
        return all_sections
    # else:
    #     split_candidate = BLOCK_SPLIT_CANDIDATES[split_candidate_index]

    chunks = []
    add_index = 0
    for add_index, split_candidate in enumerate(
        BLOCK_SPLIT_CANDIDATES[split_candidate_index:]
    ):
        chunks = re.split(split_candidate, markdown)
        if len(chunks) > 1:
            break

    for i, chunk in enumerate(chunks):
        level = split_candidate_index + add_index

        # First chunk is left-over from a previous chunk
        if i > 0:
            level += 1

        prefix = "\n\n" + "#" * level + " "
        if not chunk.strip():
            continue

        if len(chunk) <= max_chunk_size:
            all_sections.append(MarkdownChunk(string=prefix + chunk, level=level - 1))
        else:
            get_logical_blocks_recursively(
                chunk,
                max_chunk_size,
                all_sections,
                split_candidate_index=split_candidate_index + add_index + 1,
            )
    return all_sections


def markdown_splitter(
    path: Union[str, Path], max_chunk_size: int, **additional_splitter_settings
) -> List[dict]:
    """Logical split based on top-level headings.

    Args:
        markdown (str): markdown string
        max_chunk_size (int): Maximum chunk size
    """

    try:
        with open(path, "r") as f:
            markdown = f.read()
    except OSError:
        return []

    if len(markdown) < max_chunk_size:
        return [{"text": markdown, "metadata": {"heading": ""}}]

    sections = [MarkdownChunk(string="", level=0)]

    markdown, additional_metadata = preprocess_markdown(
        markdown, additional_splitter_settings
    )

    # Split by code and non-code
    chunks = markdown.split("```")

    for i, chunk in enumerate(chunks):
        if i % 2 == 0:  # Every even element (0 indexed) is a non-code
            logical_blocks = get_logical_blocks_recursively(
                chunk, max_chunk_size=max_chunk_size, all_sections=[]
            )
            sections += logical_blocks
        else:  # Process the code section
            rows = chunk.split("\n")
            code = rows[1:]

            lang = rows[0]  # Get the language name

            # Provide a hint to LLM
            all_code_rows = (
                [
                    f"\nFollowing is a code section in {lang}, delimited by triple backticks:",
                    f"```{lang}",
                ]
                + code
                + ["```"]
            )
            all_code_str = "\n".join(all_code_rows)

            # Merge code to a previous logical block if there is enough space
            if len(sections[-1].string) + len(all_code_str) < max_chunk_size:
                sections[-1] = MarkdownChunk(
                    string=sections[-1].string + all_code_str, level=sections[-1].level
                )

            # If code block is larger than max size, physically split it
            elif len(all_code_str) >= max_chunk_size:
                code_chunks = phsyical_split(
                    all_code_str, max_chunk_size=max_chunk_size
                )
                for cchunk in code_chunks:
                    # Assign language header to the code chunk, if doesn't exist
                    if f"```{lang}" not in cchunk:
                        cchunk_rows = cchunk.split("```")
                        cchunk = f"```{lang}\n" + cchunk_rows[1] + "```"

                    sections.append(
                        MarkdownChunk(string=cchunk, level=CODE_BLOCK_LEVEL)
                    )

            # Otherwise, add as a single chunk
            else:
                sections.append(
                    MarkdownChunk(string=all_code_str, level=CODE_BLOCK_LEVEL)
                )

    all_out = postprocess_sections(
        sections,
        max_chunk_size,
        additional_splitter_settings,
        additional_metadata,
        path,
    )

    logger.info(f"Got {len(all_out)} text chunks:")
    for s in all_out:
        logger.info(f"\tChunk length: {len(s['text'])}")
    return all_out


def preprocess_markdown(markdown: str, additional_settings: dict) -> Tuple[str, dict]:
    """Perform pre-processing, before splitting the document into logical blocks

    Args:
        markdown (str): string containing the document
        additional_settings (dict): Dictionary of additional pre-processing settings:
                                    remove_images (bool) - removes image links the document
                                    remove_extra_newlines (bool) - removes extra new lines (3 or more)

    Returns:
        str, dict: processed markdown string, dictionary with additional metadata
    """

    preprocess_remove_images = additional_settings.get("remove_images", False)
    preprocess_remove_extra_newlines = additional_settings.get(
        "remove_extra_newlines", True
    )
    preprocess_find_metadata = additional_settings.get("find_metadata", dict())

    if preprocess_remove_images:
        markdown = remove_images(markdown)

    if preprocess_remove_extra_newlines:
        markdown = remove_extra_newlines(markdown)

    additional_metadata = {}

    if preprocess_find_metadata:
        if not isinstance(preprocess_find_metadata, dict):
            raise TypeError(
                f"find_metadata settings should be of type dict. Got {type(preprocess_find_metadata)}"
            )

        for label, search_string in preprocess_find_metadata.items():
            logger.info(f"Looking for metadata: {search_string}")
            metadata = find_metadata(markdown, search_string)
            if metadata:
                logger.info(f"\tFound metadata for {label} - {metadata}")
                additional_metadata[label] = metadata

    return markdown, additional_metadata


def postprocess_sections(
    sections: List[MarkdownChunk],
    max_chunk_size: int,
    additional_settings: dict,
    additional_metadata: dict,
    path: Union[str, Path],
) -> List[dict]:
    all_out = []

    skip_first = additional_settings.get("skip_first", False)
    merge_headers = additional_settings.get("merge_sections", False)

    # Remove all empty sections
    sections = [s for s in sections if s.string]

    if sections and skip_first:
        logger.info("Removing first section ...")
        sections = sections[1:]

    if sections and merge_headers:
        logger.info("Merging sections ...")
        sections = merge_sections(sections, max_chunk_size=max_chunk_size)

    current_heading = ""

    sections_metadata = {"Document name": Path(path).name}

    for s in sections:
        stripped_string = s.string.strip()
        doc_metadata = {}
        if len(stripped_string) > 0:
            heading = ""
            if stripped_string.startswith("#"):  # heading detected
                heading = stripped_string.split("\n")[0].replace("#", "").strip()
                stripped_heading = heading.replace("#", "").replace(" ", "").strip()
                if not stripped_heading:
                    heading = ""
                if s.level == 0:
                    current_heading = heading
                doc_metadata["heading"] = urllib.parse.quote(
                    heading
                )  # isolate the heading
            else:
                doc_metadata["heading"] = ""

            final_section = add_section_metadata(
                stripped_string,
                section_metadata={
                    **sections_metadata,
                    **{"Subsection of": current_heading},
                    **additional_metadata,
                },
            )
            all_out.append({"text": final_section, "metadata": doc_metadata})
    return all_out


def remove_images(page_md: str) -> str:
    logger.info("Removing images ... ")
    return re.sub(r"""!\[[^\]]*\]\((.*?)\s*("(?:.*[^"])")?\s*\)""", "", page_md)


def remove_extra_newlines(page_md) -> str:
    """Remove extra new lines from the text if there are 3 or more consecutive new lines

    Args:
        page_md (str): Markdown page in a string fromate

    Returns:
        str: Markdown string with nre lines removed
    """

    logger.info("Removing extra new lines ... ")
    page_md = re.sub(r"\n{3,}", "\n\n", page_md)
    return page_md


def add_section_metadata(s, section_metadata: dict):
    metadata_s = ""
    for k, v in section_metadata.items():
        if v:
            metadata_s += f"{k}: {v}\n"
    metadata = f"Metadata applicable to the next chunk of text delimited by five stars:\n>> METADATA START\n{metadata_s}>> METADATA END\n\n"

    return metadata + "*****\n" + s + "\n*****"


def find_metadata(page_md: str, search_string: str) -> str:
    pattern = rf"{search_string}(.*)"
    match = re.search(pattern, page_md)
    if match:
        return match.group(1)
    return ""


def merge_sections(
    sections: List[MarkdownChunk], max_chunk_size: int
) -> List[MarkdownChunk]:
    """Merges sections together, up to a max chunk size"""

    current_section = sections[0]
    all_out = []

    prev_level = 0
    for s in sections[1:]:
        if (
            len(current_section.string + s.string) > max_chunk_size
            or s.level <= prev_level
        ):
            all_out.append(current_section)
            current_section = s
            prev_level = 0
        else:
            current_section = MarkdownChunk(
                string=current_section.string + s.string, level=current_section.level
            )
            prev_level = s.level if s.level != CODE_BLOCK_LEVEL else prev_level

    all_out.append(current_section)

    return all_out


if __name__ == "__main__":
    from pathlib import Path

    path = Path(
        "/storage/gdp-platform-docs/docs/platform/how-to/onboard/get-onboard-data-platform.md"
    )
    # path = Path("/storage/gdp-platform-docs/docs/platform/how-to/release/synapse_cicd_release.md")
    # path = Path("/storage/gdp-platform-docs/docs/platform/how-to/connect/get-started-azure-data-studio.md")

    print("**************************************************")
    # chunks = get_logical_blocks_recursively(text, all_sections = [], max_chunk_size=1024)
    chunks = markdown_splitter(
        path=path,
        max_chunk_size=1024,
        **{
            "merge_sections": True,
            "skip_first": True,
            "remove_images": True,
            "find_metadata": {"description": "description: "},
        },
    )
    print(len(chunks))

    for chunk in chunks:
        print("\n\nSTART CHUNK ----------------")
        print(chunk["text"])
        print("END CHUNK ----------------")

    for chunk in chunks:
        print(len(chunk["text"]))
