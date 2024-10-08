from unittest.mock import MagicMock

import pytest

from llmsearch.parsers.tables.generic import \
    pdf_table_splitter  # Replace with the actual module name


@pytest.fixture
def setup_parsed_table():
    """Fixture to create a mock parsed table for testing."""
    dummy_bbox = (0.0, 0.0, 100.0, 100.0)  # Dummy bounding box
    parsed_table = MagicMock()
    parsed_table.page_num = 1
    parsed_table.bbox = dummy_bbox
    parsed_table.caption = ""
    return parsed_table

def test_basic_functionality(setup_parsed_table):
    parsed_table = setup_parsed_table
    parsed_table.xml = [
        "<row><col name=\"A\">1</col></row>",
        "<row><col name=\"B\">2</col></row>"
    ]
    expected_output = [
        {
            "text": "```xml table:\n<row><col name=\"A\">1</col></row>\n<row><col name=\"B\">2</col></row>\n```",
            "metadata": {"page": 1, "source_chunk_type": "table"}
        }
    ]
    
    result = pdf_table_splitter(parsed_table, max_size=100)  # Adjust max size as needed
    print(result)
    assert result == expected_output

def test_caption_inclusion(setup_parsed_table):
    parsed_table = setup_parsed_table
    parsed_table.xml = ["<row><col name=\"A\">1</col></row>"]
    parsed_table.caption = "This is a test caption."
    
    expected_output = [
        {
            "text": "Table below contains information about: This is a test caption.\n```xml table:\n<row><col name=\"A\">1</col></row>\n```",
            "metadata": {"page": 1, "source_chunk_type": "table"}
        }
    ]
    
    result = pdf_table_splitter(parsed_table, max_size=100)
    assert result == expected_output

def test_caption_trimming(setup_parsed_table):
    parsed_table = setup_parsed_table
    parsed_table.xml = ["<row><col name=\"A\">1</col></row>"]
    parsed_table.caption = "A very long caption that exceeds the size limit."
    
    expected_output = [
        {
            "text": "Table below contains information about: A very long capt\n```xml table:\n<row><col name=\"A\">1</col></row>\n```",
            "metadata": {"page": 1, "source_chunk_type": "table"}
        }
    ]
    
    result = pdf_table_splitter(parsed_table, max_size=50, max_caption_size_ratio=3)
    print(result)
    assert result == expected_output

def test_element_larger_than_max_size(setup_parsed_table):
    parsed_table = setup_parsed_table
    parsed_table.xml = [
        "<row><col name=\"A\">1</col></row>",
        "<row><col name=\"B\">2</col></row>"
    ]
    long_element = "<row>" + "<col name=\"C\">" + "X" * 200 + "</col></row>"  # Very long element
    parsed_table.xml.append(long_element)
    
    result = pdf_table_splitter(parsed_table, max_size=100)
    print(result)
    # There should be one chunk for the first two elements and a separate chunk for the long element
    assert len(result) == 3

def test_empty_input(setup_parsed_table):
    parsed_table = setup_parsed_table
    parsed_table.xml = []
    parsed_table.caption = ""
    
    result = pdf_table_splitter(parsed_table, max_size=100)
    print(result)
    assert result == [
        {
            "text": "```xml table:\n```",
            "metadata": {"page": 1, "source_chunk_type": "table"}
        }
    ]

def test_single_element(setup_parsed_table):
    parsed_table = setup_parsed_table
    parsed_table.xml = ["<row><col name=\"A\">1</col></row>"]
    
    result = pdf_table_splitter(parsed_table, max_size=150)
    assert len(result) == 1

def test_multiple_elements_within_limit(setup_parsed_table):
    parsed_table = setup_parsed_table
    parsed_table.xml = [
        "<row><col name=\"A\">1</col></row>",
        "<row><col name=\"B\">2</col></row>"
    ]
    result = pdf_table_splitter(parsed_table, max_size=250)
    assert len(result) == 1