import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from checkpoint_utils import check_checkpoint, SUITE_REPOS


class CheckpointChecks(unittest.TestCase):
    def fixture(self, path):
        for name in ('config.json', 'dataset_statistics.json', 'preprocessor_config.json',
                     'tokenizer_config.json', 'tokenizer.json', 'action_head--150000_checkpoint.pt',
                     'proprio_projector--150000_checkpoint.pt', 'model-1.safetensors'):
            (path / name).write_text('{}')
        (path/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'x':'model-1.safetensors'}}))
        (path/'download-manifest.json').write_text(json.dumps({'downloaded':True, 'sha256_verified':True,
            'repo_id':SUITE_REPOS['libero_spatial'], 'revision':'abc',
            'file_metadata':[{'name':'model-1.safetensors','size':2}]}))

    def test_verified_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); self.fixture(p)
            self.assertTrue(check_checkpoint(p)['complete'])

    def test_missing_shard_and_truncation_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); self.fixture(p)
            (p/'model-1.safetensors').write_text('truncated')
            self.assertFalse(check_checkpoint(p)['complete'])
            (p/'model-1.safetensors').unlink()
            self.assertFalse(check_checkpoint(p)['complete'])

    def test_no_provenance_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); self.fixture(p)
            (p/'download-manifest.json').unlink()
            self.assertFalse(check_checkpoint(p)['complete'])

    def test_suite_ids(self):
        self.assertEqual(SUITE_REPOS['libero_10'], 'moojink/openvla-7b-oft-finetuned-libero-10')
