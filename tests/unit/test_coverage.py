import json
import pytest
from gloggur.models import Symbol
from gloggur.parsers.coverage import CoverageIngester
from gloggur.storage.metadata_store import MetadataStore

class MockMetadataStore:
    def __init__(self, symbols):
        self.symbols = symbols

    def filter_symbols(self, file_path=None, **kwargs):
        if file_path:
            return [s for s in self.symbols if s.file_path == file_path]
        return self.symbols

def test_coverage_ingester(tmp_path):
    # Setup mock symbols
    sym1 = Symbol(
        id="test.py:1:func1",
        name="func1",
        kind="function",
        file_path="test.py",
        start_line=5,
        end_line=10,
        body_hash="hash1",
        covered_by=[]
    )
    sym2 = Symbol(
        id="test.py:12:func2",
        name="func2",
        kind="function",
        file_path="test.py",
        start_line=15,
        end_line=20,
        body_hash="hash2",
        covered_by=[]
    )
    
    mock_store = MockMetadataStore([sym1, sym2])
    ingester = CoverageIngester(mock_store) # type: ignore
    
    # Create mock coverage JSON
    coverage_data = {
        "test_func1": {
            "test.py": [6, 7]
        },
        "test_integration": {
            "test.py": [9, 16]
        }
    }
    
    cov_file = tmp_path / "coverage.json"
    with open(cov_file, "w") as f:
        json.dump(coverage_data, f)
        
    # Ingest
    report = ingester.ingest_json(str(cov_file))
    
    assert report["tests_processed"] == 2
    assert report["files_affected"] == 1
    
    updated_symbols = report["symbols_to_update"]
    assert len(updated_symbols) == 2
    
    # Verify relations
    for s in updated_symbols:
        s.covered_by.sort()
        
    updated_sym1 = next(s for s in updated_symbols if s.id == sym1.id)
    assert updated_sym1.covered_by == ["test_func1", "test_integration"]
    
    updated_sym2 = next(s for s in updated_symbols if s.id == sym2.id)
    assert updated_sym2.covered_by == ["test_integration"]
