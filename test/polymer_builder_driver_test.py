import os
import pytest
from nemd import testutils

import polymer_builder_driver as driver

BASE_DIR = testutils.test_file('polym_builder')


class TestTransConformer(object):

    @pytest.fixture
    def raw_conf(self):
        polym = driver.Conformer.read(os.path.join(BASE_DIR, 'polym.sdf'))
        original_cru_mol = driver.Conformer.read(
            os.path.join(BASE_DIR, 'original_cru_mol.sdf'))
        raw_conf = driver.Conformer(polym, original_cru_mol)
        raw_conf.relax_dir = os.path.join(BASE_DIR, raw_conf.relax_dir)
        return raw_conf

    def testFoldPolym(self, raw_conf):
        raw_conf.setCruMol()
        raw_conf.setCruBackbone()
        raw_conf.foldPolym()
