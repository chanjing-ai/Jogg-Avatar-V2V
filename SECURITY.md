# Security and Release Checklist

Before publishing a Git or Hugging Face revision:

1. Review `git status` and the complete staged diff.
2. Run `rg` for tokens, credentials, private hosts, email addresses, and absolute paths.
3. Do not publish training manifests, source media, voices, face metadata, logs, or caches without explicit rights and consent.
4. Keep model weights out of GitHub. Upload only the reviewed `config.json`, model card, and exported safetensors to Hugging Face.
5. Inspect safetensors metadata and tensor keys. Do not upload pickle-based checkpoints from untrusted sources.
6. Verify the licenses and acceptable-use terms of the base model, detector, datasets, and third-party code.

Report security issues privately to the repository maintainers rather than opening a public issue.

