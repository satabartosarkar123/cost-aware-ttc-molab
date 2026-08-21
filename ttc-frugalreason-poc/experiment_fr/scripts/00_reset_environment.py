import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi

def reset_environment():
    base_dir = Path(__file__).resolve().parent.parent
    
    # 1. Clear local results
    results_dir = base_dir / "results"
    if results_dir.exists():
        print(f"Deleting local results directory: {results_dir}")
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print("Created fresh local results directory.")
    
    # 2. Clear SQLite databases in base and notebook dir
    notebook_dir = base_dir.parent.parent
    db_paths = [
        base_dir / "block_a_checkpoint.db",
        base_dir / "block_b_checkpoint.db",
        notebook_dir / "block_a_checkpoint.db",
        notebook_dir / "block_b_checkpoint.db"
    ]
    for p in db_paths:
        if p.exists():
            print(f"Deleting DB checkpoint: {p}")
            os.remove(p)
            
    # 3. Clear HuggingFace repo
    print("\nAttempting to clear Hugging Face repository...")
    try:
        import os
        token = os.environ.get("HF_TOKEN")
        if not token: raise ValueError("HF_TOKEN env var not set")
        api = HfApi(token=token)
        repo_id = "Satabarto/Molab_Checkpoints_Cost_AWARE"
        # List files and delete anything in checkpoints/
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        checkpoint_files = [f for f in files if f.startswith("checkpoints/")]
        
        if checkpoint_files:
            print(f"Found {len(checkpoint_files)} files in HF checkpoints/. Deleting...")
            # We can delete them by deleting the folder structure or files
            api.delete_folder(path_in_repo="checkpoints", repo_id=repo_id, repo_type="dataset")
            print("Successfully cleared Hugging Face repository.")
        else:
            print("Hugging Face repository already clean.")
    except Exception as e:
        print(f"Could not clear Hugging Face repository (this is non-fatal). Error: {e}")

if __name__ == "__main__":
    confirm = input("This will DESTROY all current local logs, databases, and HuggingFace checkpoints. Are you sure? (y/n): ")
    if confirm.lower() == 'y':
        reset_environment()
        print("Reset complete.")
    else:
        print("Aborted.")
