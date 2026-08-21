import json
from pathlib import Path

nb_path = Path('molab_run.ipynb')
nb = json.load(nb_path.open(encoding='utf-8'))

for c in nb['cells']:
    if c['cell_type'] == 'code':
        source = ''.join(c['source'])
        if 'GDRIVE_FILE_ID' in source and 'gdown' in source:
            # Replace google auth install with huggingface_hub
            source = source.replace(
                'pip install -q google-auth google-auth-httplib2 google-api-python-client',
                'pip install -q huggingface_hub'
            )
            # Also handle the old oauthlib variant if still there
            source = source.replace(
                'pip install -q google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client',
                'pip install -q huggingface_hub'
            )
            c['source'] = [source]
            print("Updated Cell 1 deps")
            break

nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print("Notebook saved")
