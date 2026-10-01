from pathlib import Path


def test_secret_files_are_ignored_and_example_has_placeholders() -> None:
    root = Path(__file__).parents[1]
    assert ".env" in (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    example = (root / ".env.example").read_text(encoding="utf-8")
    assert example == "ENTSOE_API_TOKEN=\n"
