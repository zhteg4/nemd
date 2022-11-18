import os
import sys
import fileutils
import pytest
import testutils
from rdkit import Chem

from unittest import mock
import polymer_builder_driver as driver
BASE_DIR = testutils.test_file('polym_builder')


class TestTransConformer(object):

    @pytest.fixture
    def raw_conf(self):
        polym = driver.TransConformer.read(os.path.join(BASE_DIR, 'polym.sdf'))
        original_cru_mol = driver.TransConformer.read(os.path.join(BASE_DIR, 'original_cru_mol.sdf'))
        raw_conf = driver.TransConformer(polym, original_cru_mol)
        raw_conf.relax_dir = os.path.join(BASE_DIR, raw_conf.relax_dir)
        return raw_conf

    def testFoldPolym(self, raw_conf):
        raw_conf.setCruMol()
        raw_conf.setBackbone()
        raw_conf.foldPolym()
        import pdb;
        pdb.set_trace()
        return
        # assert panel.log_file is None
        # with mock.patch.object(gui, 'os') as os_mock:
        #     os.path.isfile.return_value=True
        #     with mock.patch.object(gui, 'QtWidgets') as dlg_mock:
        #         panel.setLogFilePath(None)
        #         assert dlg_mock.QFileDialog.called is True
        #     assert panel.log_file is not None
        #     panel.setLogFilePath('afilename')
        #     assert panel.log_file == 'afilename'