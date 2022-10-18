from rdkit import Chem
mol1 = Chem.MolFromSmiles('*CC*')
mol2 = Chem.MolFromSmiles('C(=O)O')
combo = Chem.CombineMols(mol1, mol2)
edcombo = Chem.EditableMol(combo)
edcombo.AddBond(1, 4, order=Chem.rdchem.BondType.SINGLE)
back = edcombo.GetMol()
print(Chem.MolToSmiles(back))
