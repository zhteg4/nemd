import pytest
from nemd import structutils
from nemd import rdkitutils

METHANE = 'C'
ETHANE = 'CC'
HEXANE = 'CCCCCC'
ISOHEXANE = 'CCCC(C)C'
BENZENE = 'C1=CC=CC=C1'


class TestGriddedConf:

    def testCentroid(self, conf):
        assert np.average(conf.centroid()) == 0

    def testTranslate(self, conf):
        conf.translate([1, 2, 3])
        np.testing.assert_array_equal(conf.centroid(), [1, 2, 3])

    def testSetBondLength(self, conf):
        xyz = np.array([x * 0.1 for x in range(15)]).reshape(-1, 3)
        conf.setPositions(xyz)
        conf.setBondLength((0, 1), 2)
        np.testing.assert_almost_equal(
            rdMolTransforms.GetBondLength(conf, 0, 1), 2)


class TestFunction:

    @pytest.mark.parametrize(('smiles_str', 'nnode', 'nedge'),
                             [(METHANE, 1, 0), (ETHANE, 2, 1), (HEXANE, 6, 5),
                              (ISOHEXANE, 6, 5), (BENZENE, 6, 6)])
    def testGetGraph(self, smiles_str, nnode, nedge):
        mol = rdkitutils.get_mol_from_smiles(smiles_str)
        graph = structutils.getGraph(mol)
        assert nnode == len(graph.nodes)
        assert nedge == len(graph.edges)
