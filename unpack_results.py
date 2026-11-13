"""Restore saved predictions without network access; refuse mismatched existing files."""
from pathlib import Path
import gzip

root = Path(__file__).resolve().parent
for source in sorted((root / 'results').glob('*/answers.jsonl.gz')):
    target = source.with_suffix('')
    data = gzip.decompress(source.read_bytes())
    if target.exists() and target.read_bytes() != data:
        raise SystemExit(f'Refusing to overwrite different contents: {target}')
    if not target.exists():
        target.write_bytes(data)
    print(f'Ready: {target.relative_to(root)}')
