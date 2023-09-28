import pandas as pd
from django.http import HttpResponseRedirect
from django.shortcuts import redirect
from django.shortcuts import render
from .forms import MoleculeForm
from .models import Molecule
import subprocess



def index(request):
    if request.method == 'POST':
        form = MoleculeForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return HttpResponseRedirect('/Molecule/')
    else:
        form = MoleculeForm()
    return render(request, 'index.html', {'form': form, })


# Create your views here.
def inscrip(request):
    return render(request, 'inscription.html')

def test(request):
    return render(request, 'test_data.html')


def run_app1(request):
    command = "streamlit run app1.py"
    subprocess.Popen(command, shell=True)

    return render(request, 'index.html')

def run_app2(request):
    command = "streamlit run app2.py"
    subprocess.Popen(command, shell=True)

    return render(request,'index.html')
def run_app3(request):
    command = "streamlit run app3.py"
    subprocess.Popen(command, shell=True)

    return render(request, 'index.html')
def run_app4(request):
    command = "streamlit run app4.py"
    subprocess.Popen(command, shell=True)

    return render(request, 'index.html')
def search_results(request):
    search_query = request.GET.get('search_query')

    if search_query:
        # Effectuer la recherche dans la base de données
        results = Molecule.objects.filter(Chemblid__icontains=search_query)
    else:
        # Si aucun terme de recherche n'est fourni, renvoyer tous les éléments
        results = Molecule.objects.all()

    return render(request, 'test_data.html', {'results': results})







def Mol(request):
    molecule = Molecule.objects.all()
    if request.method == 'POST':
        MaxPhase = request.POST.get('MaxPhase')
        MolecularWeight = request.POST.get('MolecularWeight')
        Targets = request.POST.get('Targets')
        Bioactivities = request.POST.get('Bioactivities')
        RO5Violations = request.POST.get('RO5Violations')
        HBA = request.POST.get('HBA')
        HBD = request.POST.get('HBD')
        RotatableBonds = request.POST.get('RotatableBonds')
        AlogP = request.POST.get('AlogP')
        PolarSurfaceArea = request.POST.get('PolarSurfaceArea')
        QEDWeighted = request.POST.get('QEDWeighted')
        CXAcidicpKa = request.POST.get('CXAcidicpKa')
        CXBasicpKa = request.POST.get('CXBasicpKa')
        CXLogP = request.POST.get('CXLogP')
        CXLogD = request.POST.get('CXLogD')
        AromaticRings = request.POST.get('AromaticRings')
        InorganicFlag = request.POST.get('InorganicFlag')
        HeavyAtoms = request.POST.get('HeavyAtoms')
        HBA_Lipinski = request.POST.get('HBA_Lipinski')
        HBD_Lipinski = request.POST.get('HBD_Lipinski')
        RO5Violations_Lipinski = request.POST.get('RO5Violations_Lipinski')
        MolecularWeight_Monoisotopic = request.POST.get('MolecularWeight_Monoisotopic')
        Smiles = request.POST.get('Smiles')
        Chemblid = request.POST.get('Chemblid')
        molecule = Molecule(MaxPhase=MaxPhase, MolecularWeight=MolecularWeight, Targets=Targets, Bioactivities=Bioactivities, AlogP=AlogP, PolarSurfaceArea=PolarSurfaceArea, HBA=HBA, HBD=HBD,
        RO5Violations=RO5Violations, RotatableBonds=RotatableBonds, QEDWeighted=QEDWeighted, CXAcidicpKa=CXAcidicpKa, CXBasicpKa=CXBasicpKa, CXLogP=CXLogP, CXLogD=CXLogD,
        AromaticRings=AromaticRings, InorganicFlag=InorganicFlag, HeavyAtoms= HeavyAtoms, HBA_Lipinski=HBA_Lipinski, HBD_Lipinski=HBD_Lipinski,
        RO5Violations_Lipinski=RO5Violations_Lipinski, MolecularWeight_Monoisotopic=MolecularWeight_Monoisotopic, Smiles=Smiles, Chemblid=Chemblid)
        molecule.save()
        return redirect('Molecule.html')
    else:
        return render(request, 'Molecule.html')




