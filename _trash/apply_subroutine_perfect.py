import json
import re

def patch_notebook():
    with open('molab_run.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Add Log Restore patch to Cell 2 (index 2)
    cell2 = nb['cells'][2]
    src2 = "".join(cell2['source'])
    
    restore_code = """
        # --- LOG RESTORE ---
        print("\\nRestoring previous logs from Hugging Face...")
        try:
            from huggingface_hub import snapshot_download, hf_hub_download
            import zipfile
            import shutil
            
            # 1. Restore results_sync (contains block_a_logs and early data)
            try:
                print("  Syncing results_sync directory...")
                os.makedirs("results", exist_ok=True)
                snapshot_download(
                    repo_id=HF_REPO, 
                    repo_type="dataset", 
                    token=HF_TOKEN,
                    allow_patterns=["results_sync/*"],
                    local_dir=".",
                    local_dir_use_symlinks=False
                )
                if os.path.exists("results_sync"):
                    for item in os.listdir("results_sync"):
                        src = os.path.join("results_sync", item)
                        dst = os.path.join("results", item)
                        if not os.path.exists(dst):
                            shutil.move(src, dst)
            except Exception as e:
                print(f"  Warning: could not sync results_sync: {e}")

            # 2. Restore block_b zips (Day 2/Day 3 logs)
            zips_to_restore = ["block_b_qwen15b.zip", "block_b_llama32.zip"]
            for zname in zips_to_restore:
                try:
                    print(f"  Downloading checkpoint {zname}...")
                    zpath = hf_hub_download(repo_id=HF_REPO, filename=f"checkpoints/{zname}", repo_type="dataset", token=HF_TOKEN)
                    with zipfile.ZipFile(zpath, "r") as z:
                        z.extractall("results")
                    print(f"  Restored {zname}!")
                except Exception as e:
                    print(f"  Warning: could not restore {zname}: {e}")
                    
            print("Log restore complete! All past data is loaded.")
        except Exception as e:
            print(f"Log restore failed: {e}")
"""

    if "# --- LOG RESTORE ---" not in src2:
        new_src2 = re.sub(
            r'(    if all_ok:\n        print\("\\nAll files present and verified!"\)\n    else:\n.*?        for p in sorted\(base_dir\.iterdir\(\)\):\n            print\(f"  \{p\.name\}\{\'/\' if p\.is_dir\(\) else \'\'\}"\)\n)',
            r'\1\n' + restore_code,
            src2,
            flags=re.DOTALL
        )
        cell2['source'] = [l + '\n' for l in new_src2.split('\n')]
        cell2['source'][-1] = cell2['source'][-1].rstrip('\n')

    # 2. Wrap all execution cells (index >= 7) in subroutines to prevent variable leakage (p, df, fig, etc.)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        # Cells 1, 2, 3, 4 are setup cells (Standard Imports, Github/HF setup, Ollama config, Third-party imports). 
        # They MUST remain in the global scope so os, sys, and OLLAMA_MODEL are accessible.
        if i < 7:
            continue
            
        src = "".join(cell['source']).strip()
        if not src:
            continue
            
        # Avoid double-wrapping
        if f"def execute_cell_{i}()" in src:
            continue
            
        # Fix Jupyter magic commands: if the first line is a magic command, move it outside the function
        magic_lines = []
        regular_lines = []
        for line in src.split('\n'):
            if line.startswith('%') or line.startswith('!'):
                magic_lines.append(line)
            else:
                regular_lines.append(line)
                
        magic_src = "\n".join(magic_lines) + ("\n" if magic_lines else "")
        regular_src = "\n".join(regular_lines)
        
        wrapper_start = f"def execute_cell_{i}():\n"
        indented_src = "\n".join(["    " + line if line else "" for line in regular_src.split("\n")])
        wrapper_end = f"\n\nexecute_cell_{i}()"
        
        final_src = magic_src + wrapper_start + indented_src + wrapper_end
        cell['source'] = [l + '\n' for l in final_src.split('\n')]
        cell['source'][-1] = cell['source'][-1].rstrip('\n')

    with open('molab_run.ipynb', 'w', encoding='utf-8') as f:
        # Important: ensure_ascii=False ensures Greek letters stay raw in JSON (prevents UnicodeEncodeError in Marimo)
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    print("Notebook patched cleanly with perfect subroutine wrapping!")

if __name__ == "__main__":
    patch_notebook()
