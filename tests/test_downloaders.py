import sys
from pathlib import Path

from ppr_graphrag.downloaders.datasets import download_file, get_dataset_files, infer_filename_from_url


def test_get_dataset_files_hotpotqa_dev_distractor() -> None:
    files = get_dataset_files("hotpotqa", "dev_distractor")
    assert files[0]["filename"] == "hotpot_dev_distractor_v1.json"
    assert "hotpot_dev_distractor_v1.json" in files[0]["url"]


def test_infer_filename_from_manual_url() -> None:
    assert infer_filename_from_url("https://example.com/path/twowiki.zip?dl=1") == "twowiki.zip"


def test_download_file_with_mocked_requests(tmp_path, monkeypatch) -> None:
    class FakeResponse:
        headers = {"content-length": "11"}

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            assert chunk_size == 1_048_576
            yield b"hello "
            yield b"world"

    class FakeRequests:
        RequestException = RuntimeError

        @staticmethod
        def get(url, stream, timeout):
            assert url == "https://example.com/file.txt"
            assert stream is True
            assert timeout == 60
            return FakeResponse()

    monkeypatch.setitem(sys.modules, "requests", FakeRequests)
    output = tmp_path / "file.txt"
    report = download_file("https://example.com/file.txt", output)
    assert Path(report["path"]) == output
    assert output.read_text(encoding="utf-8") == "hello world"
