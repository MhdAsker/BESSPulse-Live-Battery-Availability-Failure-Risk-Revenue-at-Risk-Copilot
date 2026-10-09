from pathlib import Path


def test_secret_files_are_ignored_and_example_has_placeholders() -> None:
    root = Path(__file__).parents[1]
    assert ".env" in (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    example = (root / ".env.example").read_text(encoding="utf-8")
    lines = dict(line.split("=", 1) for line in example.splitlines())
    for secret_name in (
        "ENTSOE_API_TOKEN",
        "DATABASE_URL",
        "DATABASE_URL_TEST",
        "GEMINI_API_KEY",
        "MLFLOW_TRACKING_URI",
    ):
        assert lines[secret_name] == ""
    assert "SUPABASE_ACCESS_TOKEN" not in lines
