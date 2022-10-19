from rdkit import Chem
mol1 = Chem.MolFromSmiles('*CC*')
mol2 = Chem.MolFromSmiles('C(=O)O')
combo = Chem.CombineMols(mol1, mol2)
edcombo = Chem.EditableMol(combo)
edcombo.AddBond(1, 4, order=Chem.rdchem.BondType.SINGLE)
back = edcombo.GetMol()
print(Chem.MolToSmiles(back))

# OPLSAA Moltemplate
"""
    set type @atom:1 charge -0.22  # "Fluoride -CH2-F (UA)"    DON'T USE(OPLSUA)
    set type @atom:2 charge 0.22  # "Fluoride -CH2-F (UA)"     DON'T USE(OPLSUA)
    set type @atom:3 charge 0.55  # "Acetic Acid -COOH (UA)"   DON'T USE(OPLSUA)
    set type @atom:4 charge -0.5  # "Acetic Acid >C=O (UA)"    DON'T USE(OPLSUA)
    set type @atom:5 charge -0.58  # "Acetic Acid -OH (UA)"    DON'T USE(OPLSUA)
    set type @atom:6 charge 0.08  # "Acetic Acid CH3- (UA)"    DON'T USE(OPLSUA)
    set type @atom:7 charge 0.45  # "Acetic Acid -OH (UA)"     DON'T USE(OPLSUA)
    set type @atom:8 charge 0.0  # "Methane CH4 (UA)"          DON'T USE(OPLSUA)
    set type @atom:9 charge 0.0  # "Ethane CH3- (UA)"          DON'T USE(OPLSUA)
    set type @atom:10 charge 0.0  # "N-Alkane CH3- (UA)"       DON'T USE(OPLSUA)
    set type @atom:11 charge 0.0  # "Isobutane CH3- (UA)"      DON'T USE(OPLSUA)
    set type @atom:12 charge 0.0  # "Neopentane CH3- (UA)"     DON'T USE(OPLSUA)
    set type @atom:13 charge 0.0  # "Alkanes -CH2- (UA)"       DON'T USE(OPLSUA)
    set type @atom:14 charge 0.0  # "1-Alkene CH2= (UA)"       DON'T USE(OPLSUA)
    set type @atom:15 charge 0.0  # "Isobutane CH (UA)"        DON'T USE(OPLSUA)
    set type @atom:16 charge 0.0  # "2-Alkene -CH= (UA)"       DON'T USE(OPLSUA)
    set type @atom:17 charge 0.0  # "Aromatic CH (UA)"         DON'T USE(OPLSUA)
    set type @atom:18 charge 0.0  # "Neopentane C (UA)"        DON'T USE(OPLSUA)
    set type @atom:19 charge 0.0  # "Isobutene >C= (UA)"       DON'T USE(OPLSUA)
    set type @atom:20 charge -0.7  # "Alcohol OH (UA)"         DON'T USE(OPLSUA)
    set type @atom:21 charge 0.435  # "Alcohol OH (UA)"        DON'T USE(OPLSUA)
    set type @atom:22 charge 0.265  # "Methanol CH3- (UA)"     DON'T USE(OPLSUA)
    set type @atom:23 charge 0.265  # "Ethanol -CH2OH (UA)"    DON'T USE(OPLSUA)
    set type @atom:24 charge -0.47  # "Hydrogen Sulfide H2S"   DON'T USE(OPLSUA)
    set type @atom:25 charge -0.45  # "Alkyl Sulfide RSH (UA)" DON'T USE(OPLSUA)
    set type @atom:26 charge -0.47  # "Thioether RSR (UA)"     DON'T USE(OPLSUA)
    set type @atom:27 charge -0.3  # "Disulfide RSSR (UA)"     DON'T USE(OPLSUA)
    set type @atom:28 charge 0.235  # "Hydrogen Sulfide H2S"   DON'T USE(OPLSUA)
    set type @atom:29 charge 0.27  # "Alkyl Sulfide RSH (UA)"  DON'T USE(OPLSUA)
    set type @atom:30 charge 0.18  # "Methyl Sulfide CH3 (UA)" DON'T USE(OPLSUA)
    set type @atom:31 charge 0.18  # "Alkyl Sulfide CH2 (UA)"  DON'T USE(OPLSUA)
    set type @atom:32 charge 0.235  # "Thioether CH3 (UA)"     DON'T USE(OPLSUA)
    set type @atom:33 charge 0.235  # "Thioether CH2 (UA)"     DON'T USE(OPLSUA)
    set type @atom:34 charge 0.3  # "Disulfide CH3 (UA)"       DON'T USE(OPLSUA)
    set type @atom:35 charge 0.3  # "Disulfide CH2 (UA)"       DON'T USE(OPLSUA)
    set type @atom:36 charge -0.43  # "Acetonitrile -CN (UA)"  DON'T USE(OPLSUA)
    set type @atom:37 charge 0.28  # "Acetonitrile -CN (UA)"   DON'T USE(OPLSUA)
    set type @atom:38 charge 0.15  # "Acetonitrile CH3 (UA)"   DON'T USE(OPLSUA)
    set type @atom:39 charge 0.265  # "Isopropanol >CHOH (UA)" DON'T USE(OPLSUA)
    set type @atom:40 charge 0.265  # "t-Butanol COH (UA)"     DON'T USE(OPLSUA)
    set type @atom:41 charge -0.5  # "Ether ROR (UA)"          DON'T USE(OPLSUA)
    set type @atom:42 charge 0.25  # "Ether CH3-OR (UA)"       DON'T USE(OPLSUA)
    set type @atom:43 charge 0.25  # "Ether -CH2-OR (UA)"      DON'T USE(OPLSUA)
    set type @atom:44 charge 0.5  # "Methylene Chloride (UA)"  DON'T USE(OPLSUA)
    set type @atom:45 charge -0.25  # "Methylene Chloride (UA)"DON'T USE(OPLSUA)
    set type @atom:46 charge 0.42  # "Chloroform CHCl3 (UA)"   DON'T USE(OPLSUA)
    set type @atom:47 charge -0.14  # "Chloroform CHCl3 (UA)"  DON'T USE(OPLSUA)
    set type @atom:48 charge 0.248  # "Carbon Tetrachloride"
    set type @atom:49 charge -0.062  # "Carbon Tetrachloride"
    set type @atom:50 charge 0.139  # "DMSO >S=O (UA)"         DON'T USE(OPLSUA)
    set type @atom:51 charge -0.459  # "DMSO >S=O (UA)"        DON'T USE(OPLSUA)
    set type @atom:52 charge 0.16  # "DMSO CH3- (UA)"          DON'T USE(OPLSUA)
    set type @atom:53 charge -0.5  # "DMF C=O (UA)"            DON'T USE(OPLSUA)
    set type @atom:54 charge -0.57  # "DMF CON< (UA)"          DON'T USE(OPLSUA)
    set type @atom:55 charge 0.5  # "DMF C=O (UA)"             DON'T USE(OPLSUA)
    set type @atom:56 charge 0.285  # "DMF CH3- (UA)"          DON'T USE(OPLSUA)
"""

oplsaa = """
    set type @atom:3 charge 0.55  # "Acetic Acid -COOH (UA)"   DON'T USE(OPLSUA)
    set type @atom:4 charge -0.5  # "Acetic Acid >C=O (UA)"    DON'T USE(OPLSUA)
    set type @atom:5 charge -0.58  # "Acetic Acid -OH (UA)"    DON'T USE(OPLSUA)
    set type @atom:6 charge 0.08  # "Acetic Acid CH3- (UA)"    DON'T USE(OPLSUA)
    set type @atom:7 charge 0.45  # "Acetic Acid -OH (UA)"     DON'T USE(OPLSUA)

    set type @atom:10 charge 0.0  # "N-Alkane CH3- (UA)"       DON'T USE(OPLSUA)
    set type @atom:11 charge 0.0  # "Isobutane CH3- (UA)"      DON'T USE(OPLSUA)
    set type @atom:12 charge 0.0  # "Neopentane CH3- (UA)"     DON'T USE(OPLSUA)
    set type @atom:13 charge 0.0  # "Alkanes -CH2- (UA)"       DON'T USE(OPLSUA)
    set type @atom:14 charge 0.0  # "1-Alkene CH2= (UA)"       DON'T USE(OPLSUA)
    set type @atom:15 charge 0.0  # "Isobutane CH (UA)"        DON'T USE(OPLSUA)
    set type @atom:16 charge 0.0  # "2-Alkene -CH= (UA)"       DON'T USE(OPLSUA)
    set type @atom:17 charge 0.0  # "Aromatic CH (UA)"         DON'T USE(OPLSUA)
    set type @atom:18 charge 0.0  # "Neopentane C (UA)"        DON'T USE(OPLSUA)
    set type @atom:19 charge 0.0  # "Isobutene >C= (UA)"       DON'T USE(OPLSUA)
    set type @atom:20 charge -0.7  # "Alcohol OH (UA)"         DON'T USE(OPLSUA)
    set type @atom:21 charge 0.435  # "Alcohol OH (UA)"        DON'T USE(OPLSUA)
    set type @atom:22 charge 0.265  # "Methanol CH3- (UA)"     DON'T USE(OPLSUA)
    set type @atom:23 charge 0.265  # "Ethanol -CH2OH (UA)"    DON'T USE(OPLSUA)
    set type @atom:24 charge -0.47  # "Hydrogen Sulfide H2S"   DON'T USE(OPLSUA)
    set type @atom:25 charge -0.45  # "Alkyl Sulfide RSH (UA)" DON'T USE(OPLSUA)
    set type @atom:26 charge -0.47  # "Thioether RSR (UA)"     DON'T USE(OPLSUA)
    set type @atom:27 charge -0.3  # "Disulfide RSSR (UA)"     DON'T USE(OPLSUA)
    set type @atom:28 charge 0.235  # "Hydrogen Sulfide H2S"   DON'T USE(OPLSUA)
    set type @atom:29 charge 0.27  # "Alkyl Sulfide RSH (UA)"  DON'T USE(OPLSUA)
    set type @atom:30 charge 0.18  # "Methyl Sulfide CH3 (UA)" DON'T USE(OPLSUA)
    set type @atom:31 charge 0.18  # "Alkyl Sulfide CH2 (UA)"  DON'T USE(OPLSUA)
    set type @atom:32 charge 0.235  # "Thioether CH3 (UA)"     DON'T USE(OPLSUA)
    set type @atom:33 charge 0.235  # "Thioether CH2 (UA)"     DON'T USE(OPLSUA)
    set type @atom:34 charge 0.3  # "Disulfide CH3 (UA)"       DON'T USE(OPLSUA)
    set type @atom:35 charge 0.3  # "Disulfide CH2 (UA)"       DON'T USE(OPLSUA)
    set type @atom:36 charge -0.43  # "Acetonitrile -CN (UA)"  DON'T USE(OPLSUA)
    set type @atom:37 charge 0.28  # "Acetonitrile -CN (UA)"   DON'T USE(OPLSUA)
    set type @atom:38 charge 0.15  # "Acetonitrile CH3 (UA)"   DON'T USE(OPLSUA)
    set type @atom:39 charge 0.265  # "Isopropanol >CHOH (UA)" DON'T USE(OPLSUA)
    set type @atom:40 charge 0.265  # "t-Butanol COH (UA)"     DON'T USE(OPLSUA)
    set type @atom:41 charge -0.5  # "Ether ROR (UA)"          DON'T USE(OPLSUA)
    set type @atom:42 charge 0.25  # "Ether CH3-OR (UA)"       DON'T USE(OPLSUA)
    set type @atom:43 charge 0.25  # "Ether -CH2-OR (UA)"      DON'T USE(OPLSUA)
    set type @atom:44 charge 0.5  # "Methylene Chloride (UA)"  DON'T USE(OPLSUA)
    set type @atom:45 charge -0.25  # "Methylene Chloride (UA)"DON'T USE(OPLSUA)
    set type @atom:46 charge 0.42  # "Chloroform CHCl3 (UA)"   DON'T USE(OPLSUA)
    set type @atom:47 charge -0.14  # "Chloroform CHCl3 (UA)"  DON'T USE(OPLSUA)
    set type @atom:48 charge 0.248  # "Carbon Tetrachloride"
    set type @atom:49 charge -0.062  # "Carbon Tetrachloride"
    set type @atom:50 charge 0.139  # "DMSO >S=O (UA)"         DON'T USE(OPLSUA)
    set type @atom:51 charge -0.459  # "DMSO >S=O (UA)"        DON'T USE(OPLSUA)
    set type @atom:52 charge 0.16  # "DMSO CH3- (UA)"          DON'T USE(OPLSUA)
    set type @atom:53 charge -0.5  # "DMF C=O (UA)"            DON'T USE(OPLSUA)
    set type @atom:54 charge -0.57  # "DMF CON< (UA)"          DON'T USE(OPLSUA)
    set type @atom:55 charge 0.5  # "DMF C=O (UA)"             DON'T USE(OPLSUA)
    set type @atom:56 charge 0.285  # "DMF CH3- (UA)"          DON'T USE(OPLSUA)
"""