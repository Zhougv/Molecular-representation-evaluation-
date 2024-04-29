import os, sys
import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Fingerprints import FingerprintMols
from mordred import Calculator,descriptors
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from self_function import *
from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
from mordred import Calculator,descriptors
import pubchempy
from e3fp.config.params import default_params
from e3fp.config.params import read_params
from e3fp.pipeline import fprints_from_smiles,confs_from_smiles,fprints_from_mol
from e3fp.config.params import read_params

import selfies as sf

def getFP(smiles,fingerprint):
    
    fpdict = {
        'Avalon': avalon,
        'Day_light': daylight,
        'ECFP2': ecfp2,
        'ECFP4': ecfp4,
        'ECFP6': ecfp6,
        'MACCS': maccs,
        'E3FP': e3fp,
        'Mordred': mordred,
        'Pubchem': pubchem,
        'RDK2d': rdkit2d,
        'SELFIES': selfies,
        # '3dfp': threedfp
    }
 
    if fingerprint in fpdict:
        fp_list = fpdict[fingerprint](smiles)
    else:
        return 'nan'

    return fp_list


def daylight(smiles):

    fp_list = []
    smiles_list = []
    error = []

    for i in range(len(smiles)):
        nbits = 2048
        mol=smiles_to_rdkit_mol(smiles[i])
        if mol != None:
            smiles_list.append(smiles[i])
            bv = FingerprintMols.FingerprintMol(mol)
            temp = tuple(bv.GetOnBits())
            if temp !=():
                features = np.zeros((nbits, ))
                features[np.array(temp)] = 1
                fp_list.append(features)
            else:
                # print(i)
                error.append(i)
        else:
            print(i)
            error.append(i)

    return fp_list, smiles_list


def ecfp2(smiles):

    fp_list = []
    smiles_list = []
    error = []

    for i in range(len(smiles)):
        nbits = 2048
        molecule = smiles_to_rdkit_mol(smiles[i])
        if molecule != None:
            smiles_list.append(smiles[i])
            fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 1, nBits=nbits)
            arr = np.zeros((0,), dtype=np.float64)
            DataStructs.ConvertToNumpyArray(fp,arr)
            fp_list.append(arr)
        else:
            print(i)
            error.append(i)

    return fp_list, smiles_list


def ecfp4(smiles):

    fp_list = []
    smiles_list = []
    error = []

    for i in range(len(smiles)):
        nbits = 2048
        molecule = smiles_to_rdkit_mol(smiles[i])
        if molecule != None:
            smiles_list.append(smiles[i])
            fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=nbits)
            arr = np.zeros((0,), dtype=np.float64)
            DataStructs.ConvertToNumpyArray(fp,arr)
            fp_list.append(arr)
        else:
            print(i)
            error.append(i)

    return fp_list, smiles_list


def ecfp6(smiles):

    fp_list = []
    smiles_list = []
    error = []

    for i in range(len(smiles)):
        nbits = 2048
        molecule = smiles_to_rdkit_mol(smiles[i])
        if molecule != None:
            smiles_list.append(smiles[i])
            fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 3, nBits=nbits)
            arr = np.zeros((0,), dtype=np.float64)
            DataStructs.ConvertToNumpyArray(fp,arr)
            fp_list.append(arr)
        else:
            print(i)
            error.append(i)

    return fp_list, smiles_list


def maccs(smiles):

    fp_list = []
    smiles_list = []
    error = []

    for i in range(len(smiles)):
        mol=smiles_to_rdkit_mol(smiles[i])
        if mol !=None:
            smiles_list.append(smiles[i])
            fp=MACCSkeys.GenMACCSKeys(mol)
            arr=np.zeros((0,),dtype=np.float64)
            DataStructs.ConvertToNumpyArray(fp,arr)
            fp_list.append(arr)
        else:
            print(i)
            error.append(i)

    return fp_list, smiles_list


def pubchem(smiles):

    fp_list = []
    smiles_list = []
    error = []

    for i in range(len(smiles)):
        mol=smiles_to_rdkit_mol(smiles[i])
        if mol !=None:
            smiles_list.append(smiles[i])
            features = calcPubChemFingerAll(mol)
            features = np.array(features)
            fp_list.append(features)
        else:
            error.append(i)

    return fp_list, smiles_list


def rdkit2d(smiles):

    fp_list = []
    smiles_list = []
    error = []

    for i in range(len(smiles)):
        mol = smiles_to_rdkit_mol(smiles[i])
        if mol !=None:
            smiles_list.append(smiles[i])
            s = Chem.MolToSmiles(mol, isomericSmiles=True)
            generator = rdNormalizedDescriptors.RDKit2DNormalized()
            features = np.array(generator.process(s)[1:])
            NaNs = np.isnan(features)
            features[NaNs] = 0
            fp_list.append(features)
        else:
            error.append(i)

    return fp_list, smiles_list


def mordred(smiles):

    fp_list = []
    smiles_list = []
    error = []
    
    for i in range(len(smiles)):
        mol=smiles_to_rdkit_mol(smiles[i])
        if mol !=None:
            smiles_list.append(smiles[i])
            calc=Calculator(descriptors,ignore_3D=True)
            X_mord=pd.DataFrame(calc.pandas([mol]))
            array=np.array(X_mord,dtype=np.float64)
            aa=np.squeeze(array)
            fp_list.append(aa)
        else:
            error.append(i)

    return fp_list, smiles_list


def avalon(smiles):

    fp_list = []
    smiles_list = []
    error = []

    for i in range(len(smiles)):
        mol = smiles_to_rdkit_mol(smiles[i])
        if mol != None:
            smiles_list.append(smiles[i])
            fp1 = GetAvalonFP(mol,nBits=1024)
            arr1 = np.zeros((0,), dtype=np.float64)
            DataStructs.ConvertToNumpyArray(fp1, arr1)
            fp_list.append(arr1)
        else:
            error.append(i)
        
    return fp_list, smiles_list


def smiles_to_rdkit_mol(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except ValueError:
                return None
        return mol


# def threedfp(smiles):


#     return fp_list

def e3fp(smiles):

    fp_list = []
    smiles_list = []
    error = []

    #设置指纹参数
    fprint_params = { 'bits': 4096,'first' :1}
    
    #将smiles转换成化合物名称
    def smiles_to_name(smiles):
        compounds = pubchempy.get_compounds(smiles, namespace='smiles')
        match = compounds[0]
        name=match.iupac_name
        return name

    for i in range(len(smiles)):
        try:
            smile = smiles[i]
            mol = Chem.MolFromSmiles(smile)
            if mol is  None:
                continue
            smiles_list.append(smiles[i])
            name = smiles_to_name(smile)
            fprints = fprints_from_smiles(smile, name, fprint_params=fprint_params)
            fp_folded = fprints[0].fold(bits=1024)
            f = fp_folded.to_bitstring()
            f_list = list(f)
            fp_list.append(f_list)

        except:
            pass

    return fp_list, smiles_list

def selfies(smiles):

    selfies_list = []

    # SMILES -> SELFIES
    for i in range(len(smiles)):
        try:
            smiles_sf = sf.encoder(smiles[i])  # [C][=C][C][=C][C][=C][Ring1][=Branch1]
            smiles_smi = sf.decoder(smiles_sf)  # C1=CC=CC=C1
        except sf.EncoderError:
            continue  # sf.encoder error!

        len_smiles = sf.len_selfies(smiles_sf)  # 8

        SELFIES = list(sf.split_selfies(smiles_sf))
        selfies="".join(str(x) for x in SELFIES) #将SELFIES中所有字符串拼接成一个
        selfies_list.append(selfies)
    # ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[=Branch1]']
        #print(SELFIES)

    dataset = np.array(selfies_list)

    #dataset = ["[C][O][C]", "[F][C][F]", "[O][=O]", "[C][C][O][C][C]"]
    alphabet = sf.get_alphabet_from_selfies(dataset)
    alphabet.add("[nop]")  # [nop] is a special padding symbol
    alphabet = list(sorted(alphabet))  # ['[=O]', '[C]', '[F]', '[O]', '[nop]']

    pad_to_len = max(sf.len_selfies(s) for s in dataset)  # 5
    symbol_to_idx = {s: i for i, s in enumerate(alphabet)}

    fp_list=[]
    smiles_list = []

    for i in range(len(dataset)):
        try:
            dimethyl_ether = dataset[i]
            label, one_hot = sf.selfies_to_encoding(selfies=dimethyl_ether,vocab_stoi=symbol_to_idx,pad_to_len=pad_to_len,enc_type="both")
            one_hot_1=[j for i in one_hot for j in i] #将同一list中多个list拼接
            fp_list.append(one_hot_1)
            smiles_list.append(smiles[i])

        except:
            pass

    return fp_list, smiles_list