"""DATA-019: tokenize_and_chunk must finalize without a Windows mmap lock crash.

The finalizer did `data = mmap_data[:num_chunks]` (a memmap VIEW), saved it,
`del mmap_data`, then `tmp_path.unlink()`. On Windows the view kept the temp
file mapped, so the (unguarded) unlink raised PermissionError [WinError 32] —
crashing data-prep (stage 2) at the very end on the primary platform, after the
output was already written, and leaving an orphaned _tmp.npy. (Same class as
DATA-004, which fixed the dedup path.) This test runs the full path end-to-end
and asserts it completes and cleans up the temp file. The module had zero tests.
"""

from pathlib import Path

from cola_coder.data.preprocess import tokenize_and_chunk, load_processed_data
from cola_coder.tokenizer import train_tokenizer as tt
from cola_coder.tokenizer.tokenizer_utils import CodeTokenizer


def _tokenizer(tmp_path: Path) -> CodeTokenizer:
    out = str(tmp_path / "tok.json")
    tt.train_from_iterator(
        iter(["def f():\n  return 1\n", "const x = 1;\n"] * 100),
        vocab_size=320, output_path=out,
    )
    return CodeTokenizer(out)


class TestFinalize:
    def test_completes_and_removes_temp_file(self, tmp_path):
        tok = _tokenizer(tmp_path)
        proc = tmp_path / "proc"
        texts = ["def foo():\n    return 42\n"] * 50
        # Before the fix this raised PermissionError at the unlink on Windows.
        out = tokenize_and_chunk(
            iter(texts), tok, chunk_size=8,
            output_dir=str(proc), output_name="t", batch_size=4,
        )
        arr = load_processed_data(out)
        assert arr.ndim == 2 and arr.shape[1] == 8 and arr.shape[0] > 0
        # Temp memmap file must be deleted (the unlink succeeded).
        assert not (proc / "t_tmp.npy").exists()
        # Manifest written alongside.
        assert (proc / "t.manifest.yaml").exists()

    def test_empty_iterator_no_crash(self, tmp_path):
        tok = _tokenizer(tmp_path)
        proc = tmp_path / "proc_empty"
        out = tokenize_and_chunk(
            iter([]), tok, chunk_size=8,
            output_dir=str(proc), output_name="e", batch_size=4,
        )
        # num_chunks == 0 path: still saves a placeholder and cleans up.
        assert Path(out).exists()
        assert not (proc / "e_tmp.npy").exists()
