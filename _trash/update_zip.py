import json, zipfile
from pathlib import Path

# 1. Update the notebook Cell 1 - simplify Drive auth section
nb = json.load(open('molab_run.ipynb', encoding='utf-8'))
for c in nb['cells']:
    if c['cell_type'] == 'code':
        source = ''.join(c['source'])
        if 'GDRIVE_FILE_ID' in source and 'gdown' in source:
            # Replace the old oauth install with simpler one
            source = source.replace(
                'pip install -q google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client',
                'pip install -q google-auth google-auth-httplib2 google-api-python-client'
            )
            c['source'] = [source]
            break

with open('molab_run.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook updated")

# 2. Update the zip with both files
src = Path('../Cost-Aware-Test-Time-upload.zip')
tmp = Path('../upload_new_3.zip')
replace = {'gdrive_oauth.py', 'molab_run.ipynb'}

with zipfile.ZipFile(str(src), 'r') as zin:
    with zipfile.ZipFile(str(tmp), 'w', zipfile.ZIP_DEFLATED) as zout:
        seen = set()
        for item in zin.infolist():
            if item.filename in replace or item.filename in seen:
                continue
            seen.add(item.filename)
            zout.writestr(item, zin.read(item.filename))
        zout.write('gdrive_oauth.py', 'gdrive_oauth.py')
        zout.write('molab_run.ipynb', 'molab_run.ipynb')

src.unlink()
tmp.rename(src)

with zipfile.ZipFile(str(src)) as z:
    for name in ['gdrive_oauth.py', 'molab_run.ipynb']:
        info = next(i for i in z.infolist() if i.filename == name)
        print(f"  {name}: {info.file_size} bytes")
    print(f"Total zip: {src.stat().st_size/1024/1024:.1f} MB")
print("Done!")
