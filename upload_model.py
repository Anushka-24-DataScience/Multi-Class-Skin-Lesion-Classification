from huggingface_hub import HfApi
import os

# Retrieve the Hugging Face token from an environment variable
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError(
        "HF_TOKEN not found. Please set it as an environment variable."
    )

api = HfApi()

# Upload the entire artifacts folder
api.upload_folder(
    folder_path="HF_space_deployment/artifacts",
    path_in_repo="artifacts",
    repo_id="AnushkaSrivastava/DermaCancerScan",
    repo_type="space",
    token=hf_token
)

print("Artifacts uploaded successfully!")




