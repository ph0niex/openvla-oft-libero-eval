import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from checkpoint import verify_files
from checkpoint_utils import SUITE_REPOS


class OfficialProfiles(unittest.TestCase):
    def test_profiles_match_pinned_runner_defaults(self):
        source=(ROOT/'vendor/openvla-oft/experiments/robot/libero/run_libero_eval.py').read_text()
        config=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='GenerateConfig')
        defaults={}
        for n in config.body:
            if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name):
                try: defaults[n.target.id]=ast.literal_eval(n.value)
                except (ValueError,TypeError): pass
        for suite,repo in SUITE_REPOS.items():
            p=json.loads((ROOT/f'profiles/{suite}.l1.official.json').read_text())
            self.assertEqual(p['checkpoint_repo'],repo)
            for name,value in p['options'].items():
                if name=='task_suite_name':
                    self.assertEqual(value,suite)
                else:
                    self.assertEqual(value,defaults[name],f'{suite}: {name}')

    def test_size_and_hash_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp); (path/'weight').write_bytes(b'good')
            item={'name':'weight','size':4,'sha256':hashlib.sha256(b'good').hexdigest()}
            self.assertFalse(verify_files(path,[item]))
            (path/'weight').write_bytes(b'evil')
            self.assertIn('SHA256',verify_files(path,[item])[0])
            (path/'weight').unlink()
            self.assertIn('Missing',verify_files(path,[item])[0])
