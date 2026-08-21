import os
from google.oauth2 import service_account
from googleapiclient.discovery import build

SA_JSON = 'molab_sa.json'
FOLDER_ID = '1LLiRgpGJ65iPuR1fCOprDVm-b_bUkdVJ'

if os.path.exists(SA_JSON):
    creds = service_account.Credentials.from_service_account_file(SA_JSON, scopes=['https://www.googleapis.com/auth/drive'])
    svc = build('drive', 'v3', credentials=creds, cache_discovery=False)
    res = svc.files().list(q=f"'{FOLDER_ID}' in parents and trashed=false", fields='files(name, createdTime)').execute()
    files = res.get('files', [])
    print(f'Found {len(files)} files in Drive folder:')
    for f in sorted(files, key=lambda x: x.get('createdTime', ''), reverse=True)[:10]:
        print(f" - {f['name']} (created {f.get('createdTime')})")
else:
    print('molab_sa.json not found')
