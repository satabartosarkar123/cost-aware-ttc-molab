import json
from pathlib import Path

nb_path = Path(r"c:\Users\USER\Cost-Aware-Test-Time\Cost-Aware-Test-time-molab\molab_run.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        
        # Fix the empty braces inside f-strings
        bad_fstring1 = "f\"python -c \\\"import gdown; gdown.upload('{}'.format(tmp_zip), folder_id='{DRIVE_FOLDER_ID}', use_cookies=False)\\\"\""
        good_fstring1 = "f\"python -c \\\"import gdown; gdown.upload('{tmp_zip}', folder_id='{DRIVE_FOLDER_ID}', use_cookies=False)\\\"\""
        
        if bad_fstring1 in source:
            source = source.replace(bad_fstring1, good_fstring1)
            
            lines = [line + "\n" for line in source.split("\n")]
            if lines: lines[-1] = lines[-1][:-1]
            cell["source"] = lines

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Notebook f-string syntax errors patched successfully.")
