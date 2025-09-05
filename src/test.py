import os
from pathlib import Path

# 1-Verifica se le variabili di progetto sono impostate e le cartelle raggiungibili
base_folder = os.environ["PROJ_BASE_DIR"]
dvc_data_folder = os.environ["DVC_DATA_DIR"]
dvc_models_folder = os.environ["DVC_MODELS_DIR"]
writer_folder = os.environ["WRITER_DIR"]

dirs_to_check = [base_folder, dvc_data_folder, dvc_models_folder,writer_folder]

for current_path in dirs_to_check:
    cartella = Path(current_path)
    if cartella.is_dir():
        print(f"OK! La cartella {cartella} esiste.")
    else:
        print(f"ERRORE: La cartella {cartella} non esiste.")


