import pytest
from gloggur.parsers.treesitter_parser import TreeSitterParser


@pytest.fixture
def python_parser():
    return TreeSitterParser("python")


def test_extract_call_graph(python_parser):
    source = """
import json
from my_module import helper

def my_function(data):
    result = helper(data)
    json.loads(result)
    print("done")
    return result
"""
    symbols = python_parser.extract_symbols("test.py", source)
    
    assert len(symbols) == 1
    func_symbol = symbols[0]
    
    expected_calls = ["helper", "json.loads", "print"]
    assert func_symbol.calls == expected_calls
