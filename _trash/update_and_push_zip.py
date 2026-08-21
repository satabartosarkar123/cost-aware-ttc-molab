import zipfile
from pathlib import Path

src = Path('../Cost-Aware-Test-Time-upload.zip')
tmp = Path('../upload_new_4.zip')
replace = {'gdrive_oauth.py', 'molab_run.ipynb', 'molab_run_fixed.ipynb', 'requirements_molab.txt'}

with zipfile.ZipFile(str(src), 'r') as zin:
    with zipfile.ZipFile(str(tmp), 'w', zipfile.ZIP_DEFLATED) as zout:
        seen = set()
        for item in zin.infolist():
            if item.filename in replace or item.filename in seen:
                continue
            seen.add(item.filename)
            zout.writestr(item, zin.read(item.filename))
        
        if Path('gdrive_oauth.py').exists():
            zout.write('gdrive_oauth.py', 'gdrive_oauth.py')
        zout.write('molab_run.ipynb', 'molab_run.ipynb')
        zout.write('molab_run_fixed.ipynb', 'molab_run_fixed.ipynb')
        zout.write('requirements_molab.txt', 'requirements_molab.txt')

src.unlink()
tmp.rename(src)
print("Zip updated!")

from huggingface_hub import HfApi
_api = HfApi(token="REDACTED")
print("Uploading to HF...")
_api.upload_file(
    path_or_fileobj=str(src),
    path_in_repo="Cost-Aware-Test-Time-upload.zip",
    repo_id="Satabarto/Molab_Checkpoints_Cost_AWARE",
    repo_type="dataset",
)
print("HF upload successful!")
