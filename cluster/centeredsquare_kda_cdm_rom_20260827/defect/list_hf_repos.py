#!/usr/bin/env python3
"""List repositories visible to the VM's existing Hugging Face login."""

from huggingface_hub import HfApi


api = HfApi()
identity = api.whoami()
name = identity["name"]
print(f"account={name}")
for kind, iterator in (
    ("dataset", api.list_datasets(author=name)),
    ("model", api.list_models(author=name)),
):
    for repository in iterator:
        print(f"{kind}={repository.id}")
