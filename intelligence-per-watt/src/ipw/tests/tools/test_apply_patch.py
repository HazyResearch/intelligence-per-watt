"""Tests for tools/apply_patch.py."""

from __future__ import annotations

import asyncio

from ipw.tools.apply_patch import ApplyPatchTool

SAMPLE_PATCH = """\
--- a/file.txt
+++ b/file.txt
@@ -1 +1 @@
-hello
+goodbye
"""


class TestApplyPatchTool:
    def test_spec(self) -> None:
        spec = ApplyPatchTool.spec
        assert spec.name == "apply_patch"
        assert spec.side_effect_conflict is True

    def test_simple_patch_applies(self, tmp_path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("hello\n")
        result = asyncio.run(ApplyPatchTool().run(patch=SAMPLE_PATCH, cwd=str(tmp_path)))
        assert result.success is True
        assert f.read_text() == "goodbye\n"

    def test_malformed_patch_rejected(self, tmp_path) -> None:
        result = asyncio.run(ApplyPatchTool().run(patch="not a patch", cwd=str(tmp_path)))
        assert result.success is False

    def test_patch_without_target_file_fails(self, tmp_path) -> None:
        result = asyncio.run(ApplyPatchTool().run(patch=SAMPLE_PATCH, cwd=str(tmp_path)))
        assert result.success is False
