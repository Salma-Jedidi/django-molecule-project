from django.forms import ModelForm
from myapp.models import Molecule


class MoleculeForm(ModelForm):
    class Meta:
        model = Molecule
        fields = ['MaxPhase', 'MolecularWeight', 'Targets', 'Bioactivities', 'AlogP', 'PolarSurfaceArea', 'HBA', 'HBD',
                  'RO5Violations', 'RotatableBonds', 'QEDWeighted', 'CXAcidicpKa', 'CXBasicpKa', 'CXLogP', 'CXLogD',
                  'AromaticRings', 'InorganicFlag', 'HeavyAtoms', 'HBA_Lipinski', 'HBD_Lipinski',
                  'RO5Violations_Lipinski', 'MolecularWeight_Monoisotopic', 'Smiles','Chemblid']
