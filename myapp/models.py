from django.db import models



class Molecule(models.Model):
    Chemblid = models.CharField(primary_key=True, max_length=100)
    MaxPhase = models.IntegerField(null=True, blank=True)
    MolecularWeight = models.FloatField()
    Targets = models.FloatField()
    Bioactivities = models.FloatField()
    AlogP = models.FloatField()
    PolarSurfaceArea = models.CharField(max_length=30)
    HBA = models.CharField(max_length=30)
    HBD = models.CharField(max_length=30)
    RO5Violations = models.CharField(max_length=30)
    RotatableBonds = models.CharField(max_length=30)
    QEDWeighted = models.CharField(max_length=30)
    CXAcidicpKa = models.CharField(max_length=30)
    CXBasicpKa = models.CharField(max_length=30)
    CXLogP = models.CharField(max_length=30)
    CXLogD = models.FloatField(max_length=30)
    AromaticRings = models.FloatField(max_length=30)
    InorganicFlag = models.IntegerField()
    HeavyAtoms = models.FloatField(max_length=30)
    HBA_Lipinski =models.FloatField(max_length=30)
    HBD_Lipinski =models.FloatField(max_length=30)
    RO5Violations_Lipinski =models.FloatField(max_length=30)
    MolecularWeight_Monoisotopic =models.FloatField(max_length=30)
    Smiles = models.CharField(max_length=30)

    class Meta:
        app_label = 'myapp'