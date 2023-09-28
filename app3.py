from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit import Chem
import streamlit as st

def cal_prop(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    return Chem.MolToSmiles(m), ExactMolWt(m), MolLogP(m), CalcTPSA(m)

def main():
    st.title("Molecular Property Calculator")

    smiles_input = st.text_input("Enter SMILES", "")
    if st.button("Calculate"):
        result = cal_prop(smiles_input)
        if result is not None:
            st.write("SMILES: ", result[0])
            st.write("Exact Molecular Weight: ", result[1])
            st.write("Molecular LogP: ", result[2])
            st.write("Topological Polar Surface Area (TPSA): ", result[3])
        else:
            st.write("Invalid SMILES input.")

if __name__ == '__main__':
    main()