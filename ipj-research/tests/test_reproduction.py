from __future__ import annotations

import pytest

from ipj.reproduction import compare_macros, sha256, tex_macros, verify_inputs


def test_tex_definitions_handle_tables_nested_braces_and_comments():
    source = r"""
    % \newcommand{\Hidden}{not a definition}
    \newcommand{\Rows}{A & {1{,}000} \\}
    \newcommand{\Rate}{86.3\%}
    \newcommand{\Escaped}{\{literal\}}
    \renewcommand{\Rate}{86.4\%} % author-approved override
    \providecommand{\Rate}{incorrect replacement}
    """
    assert tex_macros(source) == {
        "Rows": r"A & {1{,}000} \\",
        "Rate": r"86.4\%",
        "Escaped": r"\{literal\}",
    }


def test_unclosed_or_empty_macro_file_is_rejected():
    with pytest.raises(ValueError, match="Unclosed"):
        tex_macros(r"\newcommand{\Value}{1")
    with pytest.raises(ValueError, match="No paper"):
        tex_macros("% empty file")


def test_changed_missing_and_extra_values_fail_comparison():
    result = compare_macros({"Count": "89", "Rate": "86.3"}, {"Rate": "86.4", "Extra": "1"})
    assert result["status"] == "failed"
    assert result["matched_count"] == 0
    assert result["missing"] == ["Count"]
    assert result["extra"] == ["Extra"]
    assert result["changed"]["Rate"] == {"expected": "86.3", "actual": "86.4"}


def test_input_tampering_and_path_escape_are_rejected(tmp_path):
    root = tmp_path / "study"
    root.mkdir()
    data = root / "data.json"
    data.write_text('{"n": 89}')
    expected = sha256(data)
    assert verify_inputs(root, {"data.json": expected}) == []
    data.write_text('{"n": 88}')
    assert verify_inputs(root, {"data.json": expected}) == ["Input hash mismatch: data.json"]
    assert verify_inputs(root, {"../outside": expected}) == ["Input escapes artifact directory: ../outside"]
    assert verify_inputs(root, {}) == ["The input manifest is empty"]
