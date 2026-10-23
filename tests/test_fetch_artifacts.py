"""Offline publication checks: never download artifacts or call a model/API."""

import importlib.util
import io
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("fetch_artifacts", Path(__file__).resolve().parents[1] / "scripts/fetch_artifacts.py")
fetch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch)


class ArtifactRestoreTests(unittest.TestCase):
    def make_archive(self, path, items):
        with tarfile.open(path, "w:gz") as archive:
            for name, content in items:
                member = tarfile.TarInfo(name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))

    def test_restore_is_repeatable_and_conflicts_do_not_partially_write(self):
        with TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            root.mkdir()
            archive = Path(temp) / "data.tar.gz"
            self.make_archive(archive, [("data/new.txt", b"new"), ("data/existing.txt", b"expected")])
            (root / "data").mkdir()
            (root / "data/existing.txt").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "Existing file differs"):
                fetch.restore_archive(archive, root, ["data"])
            self.assertFalse((root / "data/new.txt").exists())
            (root / "data/existing.txt").write_bytes(b"expected")
            self.assertEqual(fetch.restore_archive(archive, root, ["data"]), 1)
            self.assertEqual(fetch.restore_archive(archive, root, ["data"]), 0)

    def test_path_escape_and_symlink_destinations_are_rejected(self):
        with TemporaryDirectory() as temp:
            root = Path(temp) / "project"
            root.mkdir()
            archive = Path(temp) / "data.tar.gz"
            self.make_archive(archive, [("data/../../escape.txt", b"bad")])
            with self.assertRaisesRegex(ValueError, "Unsafe artifact path"):
                fetch.restore_archive(archive, root, ["data"])
            (root / "data").symlink_to(Path(temp), target_is_directory=True)
            self.make_archive(archive, [("data/escape.txt", b"bad")])
            with self.assertRaisesRegex(ValueError, "Symlink"):
                fetch.restore_archive(archive, root, ["data"])
            self.assertFalse((Path(temp) / "escape.txt").exists())

    def test_existing_model_must_match_manifest_without_downloading(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "models/community"
            model.mkdir(parents=True)
            (model / "config.json").write_bytes(b"original")
            manifest = {"files": {"config.json": fetch.digest(model / "config.json")}}
            metadata = root / "manifest.json"
            metadata.write_text(json.dumps(manifest))
            (model / "CONVERSION_MANIFEST.json").write_bytes(metadata.read_bytes())
            item = {"destination": "models/community", "conversion_manifest": "manifest.json"}
            with patch.object(fetch, "ROOT", root), patch("huggingface_hub.snapshot_download") as download:
                fetch.restore_community_model(item)
                (model / "config.json").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "Existing model file differs"):
                    fetch.restore_community_model(item)
                download.assert_not_called()

    def test_chunked_bundle_is_verified_before_restore(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "fixture.tar.gz"
            self.make_archive(archive, [("data/fixture.txt", b"original")])
            content = archive.read_bytes()
            parts = []
            for index, block in enumerate([content[:17], content[17:]]):
                path = root / f"part-{index}"
                path.write_bytes(block)
                parts.append({"filename": path.name, "bytes": len(block), "sha256": fetch.digest(path)})
            item = {"filename": archive.name, "bytes": len(content), "sha256": fetch.digest(archive),
                    "parts": parts, "repo_id": "unused", "revision": "frozen", "allowed_prefixes": ["data"]}
            destination = root / "restored"
            destination.mkdir()
            download = lambda repo, filename, **kwargs: root / filename
            self.assertEqual(fetch.restore_bundle(item, destination, download), 1)
            self.assertEqual((destination / "data/fixture.txt").read_bytes(), b"original")
            (root / "part-1").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "Artifact part SHA256/size mismatch"):
                fetch.restore_bundle(item, root / "must-not-exist", download)
            self.assertFalse((root / "must-not-exist").exists())
