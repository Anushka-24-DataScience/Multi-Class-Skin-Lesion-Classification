import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN", "").strip()
username = os.environ.get("HF_USERNAME", "").strip()
space_name = os.environ.get("HF_SPACE_NAME", "").strip()

print(f"Username length: {len(username)}")
print(f"Space name length: {len(space_name)}")
print(f"Token starts with hf_: {token.startswith('hf_')}")
print(f"Repo id will be: {username}/{space_name}")

if not username or not space_name or not token:
    raise ValueError("One or more secrets are empty! Check GitHub secrets.")

api = HfApi()

api.upload_folder(
    folder_path="HF_space_deployment",
    repo_id=f"{username}/{space_name}",
    repo_type="space",
    token=token,
    ignore_patterns=["uploads/*", "uploads/"]
)

print(f"Successfully deployed to: https://huggingface.co/spaces/{username}/{space_name}")