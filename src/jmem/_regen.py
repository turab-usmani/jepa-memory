"""Regenerate generated tables: ``python -m jmem._regen``."""

from __future__ import annotations

from jmem import aggregate
from jmem.paths import repo_root
from jmem.utils import load_yaml


def main() -> None:
    root = repo_root()
    refs = load_yaml(root / "configs" / "eval" / "references.yaml")
    out = root / "results" / "reproduction"
    rows = aggregate.summarize(aggregate.load_results(out, pattern="official-*.json"))
    table = aggregate.reproduction_table(rows, refs["references"], refs["tolerance"])
    (out / "reproduction_table.md").write_text(table + "\n", encoding="utf-8", newline="\n")
    readme = root / "README.md"
    text = aggregate.replace_block(readme.read_text(encoding="utf-8"), "reproduction", table)
    readme.write_text(text, encoding="utf-8", newline="\n")
    print(f"regenerated reproduction table: {len(rows)} rows")


if __name__ == "__main__":
    main()
