"""Deployment archives include only verified, explicitly named model artifacts."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.package_live import model_files


class ModelPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)

    def file(self,name):
        body=('fixture:'+name).encode();(self.folder/name).write_bytes(body)
        return hashlib.sha256(body).hexdigest()

    def manifest(self,schema=2):
        models={}
        for symbol in ('BTCUSDT','ETHUSDT'):
            seed=f'{symbol}-training.csv.gz'
            item={'training_samples_file':seed,'training_samples_sha256':self.file(seed)}
            if schema==1:
                name=f'{symbol}.joblib';item.update(file=name,sha256=self.file(name))
            else:
                item['members']={}
                for key in ('logistic_1m','gbm_deeper_5m'):
                    name=f'{symbol}-{key}.joblib'
                    item['members'][key]={'file':name,'sha256':self.file(name)}
            models[symbol]=item
        result={'schema_version':schema,'models':models};self.save(result);return result

    def save(self,value):
        (self.folder/'manifest.json').write_text(json.dumps(value),encoding='utf-8')

    def test_consensus_seven_files_excludes_verification_and_arbitrary_artifacts(self):
        self.manifest();self.file('verification.json');self.file('private.joblib');self.file('extra.csv.gz')
        files=model_files(self.folder)
        self.assertEqual(len(files),7)
        self.assertNotIn('verification.json',files);self.assertNotIn('private.joblib',files)
        self.assertIn('BTCUSDT-logistic_1m.joblib',files)

    def test_legacy_five_files_remain_supported(self):
        self.manifest(1);files=model_files(self.folder)
        self.assertEqual(len(files),5);self.assertIn('ETHUSDT.joblib',files)

    def test_corrupt_seed_or_member_rejected(self):
        for name in ('BTCUSDT-training.csv.gz','BTCUSDT-gbm_deeper_5m.joblib'):
            self.manifest();(self.folder/name).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):model_files(self.folder)

    def test_path_traversal_and_incomplete_pair_rejected(self):
        value=self.manifest();value['models']['BTCUSDT']['training_samples_file']='../secret.csv.gz';self.save(value)
        with self.assertRaisesRegex(ValueError,'filename'):model_files(self.folder)
        value=self.manifest();del value['models']['ETHUSDT']['members']['logistic_1m'];self.save(value)
        with self.assertRaisesRegex(ValueError,'member'):model_files(self.folder)

    def test_unknown_schema_or_asset_rejected(self):
        value=self.manifest();value['schema_version']=3;self.save(value)
        with self.assertRaises(ValueError):model_files(self.folder)
        value=self.manifest();value['models']['SOLUSDT']=value['models'].pop('ETHUSDT');self.save(value)
        with self.assertRaises(ValueError):model_files(self.folder)


if __name__=='__main__':unittest.main()
