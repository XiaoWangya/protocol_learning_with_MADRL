import shutil, os

if len(os.listdir('./runs/')):
    for file in os.listdir('./runs/'):
        shutil.move('./runs/'+file, './his_runs/')
    