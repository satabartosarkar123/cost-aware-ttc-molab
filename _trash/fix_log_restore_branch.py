"""
Fix: Move Log Restore code out of the 'else' branch so it ALWAYS runs.
The Log Restore was accidentally nested inside 'else: # some files missing',
but it needs to run every time regardless.
"""
import json

def fix():
    with open('molab_run.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cell2 = nb['cells'][2]
    src = "".join(cell2['source'])
    
    # The problem: Log Restore is inside the else block (indented 8 spaces)
    # We need to move it out so it's at the same level as the if/else (indented 4 spaces)
    
    # Strategy: Replace the entire if/else + Log Restore block with the correct structure
    old_block = """    if all_ok:
        print("\\nAll files present and verified!")
    else:
        print("\\nSome files missing — check zip contents.")
        # Debug: show what IS in the directory
        print("\\nFiles in base_dir:")
        for p in sorted(base_dir.iterdir()):
            print(f"  {p.name}{'/' if p.is_dir() else ''}")


        # --- LOG RESTORE ---
        print("Restoring previous logs from Hugging Face...")
        try:
            from huggingface_hub import snapshot_download, hf_hub_download
            import zipfile
            import shutil
            
            # 1. Restore results_sync (contains block_a_logs and early data)
            try:
                print("  Syncing results_sync directory...")
                os.makedirs(str(base_dir / "results"), exist_ok=True)
                snapshot_download(
                    repo_id=HF_REPO, 
                    repo_type="dataset", 
                    token=HF_TOKEN,
                    allow_patterns=["results_sync/*"],
                    local_dir=str(base_dir),
                    local_dir_use_symlinks=False
                )
                if os.path.exists(str(base_dir / "results_sync")):
                    for item in os.listdir(str(base_dir / "results_sync")):
                        src = os.path.join(str(base_dir / "results_sync"), item)
                        dst = os.path.join(str(base_dir / "results"), item)
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
                        z.extractall(str(base_dir / "results"))
                    print(f"  Restored {zname}!")
                except Exception as e:
                    print(f"  Warning: could not restore {zname}: {e}")
                    
            print("Log restore complete! All past data is loaded.")
        except Exception as e:
            print(f"Log restore failed: {e}")"""

    new_block = """    if all_ok:
        print("\\nAll files present and verified!")
    else:
        print("\\nSome files missing — check zip contents.")
        # Debug: show what IS in the directory
        print("\\nFiles in base_dir:")
        for p in sorted(base_dir.iterdir()):
            print(f"  {p.name}{'/' if p.is_dir() else ''}")

    # --- LOG RESTORE (always runs) ---
    print("\\nRestoring previous logs from Hugging Face...")
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        import zipfile as _zf
        
        # 1. Restore results_sync (contains block_a_logs and early data)
        try:
            print("  Syncing results_sync directory...")
            os.makedirs(str(base_dir / "results"), exist_ok=True)
            snapshot_download(
                repo_id=HF_REPO, 
                repo_type="dataset", 
                token=HF_TOKEN,
                allow_patterns=["results_sync/*"],
                local_dir=str(base_dir),
                local_dir_use_symlinks=False
            )
            _rs = base_dir / "results_sync"
            if _rs.exists():
                for item in os.listdir(str(_rs)):
                    _src = _rs / item
                    _dst = base_dir / "results" / item
                    if not _dst.exists():
                        shutil.move(str(_src), str(_dst))
        except Exception as e:
            print(f"  Warning: could not sync results_sync: {e}")

        # 2. Restore block_b zips (Day 2/Day 3 logs)
        zips_to_restore = ["block_b_qwen15b.zip", "block_b_llama32.zip"]
        for zname in zips_to_restore:
            try:
                print(f"  Downloading checkpoint {zname}...")
                zpath = hf_hub_download(repo_id=HF_REPO, filename=f"checkpoints/{zname}", repo_type="dataset", token=HF_TOKEN)
                with _zf.ZipFile(zpath, "r") as zf:
                    zf.extractall(str(base_dir / "results"))
                print(f"  Restored {zname}!")
            except Exception as e:
                print(f"  Warning: could not restore {zname}: {e}")
                
        print("Log restore complete! All past data is loaded.")
    except Exception as e:
        print(f"Log restore failed: {e}")"""

    if old_block in src:
        src = src.replace(old_block, new_block)
        cell2['source'] = [l + '\n' for l in src.split('\n')]
        cell2['source'][-1] = cell2['source'][-1].rstrip('\n')
        print("SUCCESS: Log Restore moved out of else branch — now ALWAYS runs!")
    else:
        print("ERROR: Could not find the old block to replace!")
        # Try to find partial matches for debugging
        if "# --- LOG RESTORE ---" in src:
            print("  Log Restore header found in source")
        if "if all_ok:" in src:
            print("  'if all_ok:' found in source")

    with open('molab_run.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

if __name__ == "__main__":
    fix()
