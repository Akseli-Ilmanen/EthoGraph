# Publishing to PyPI

Publishing is automated via GitHub Actions (`.github/workflows/test_and_deploy.yml`).
It triggers on **git tags**, not on every push to main.

## One-time setup

Add your PyPI API token as a GitHub secret:
1. Go to repo → Settings → Secrets and variables → Actions
2. New repository secret: name `TWINE_API_KEY`, value = your PyPI token (starts with `pypi-`)

Store your token locally so twine never prompts interactively — create `C:\Users\<you>\.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-your-token-here
```

This file lives outside the repo and is never committed.

## Release steps

1. Ensure working tree is clean (`git status` shows nothing modified)
2. Commit and push all changes to main
3. Tag and push the tag:
   ```
   git tag v0.1.3
   git push origin v0.1.3
   ```
4. GitHub Actions runs: lint → tests → build → upload to PyPI automatically

## Manual upload (fallback)

If you need to publish without GitHub Actions:

```powershell
# Activate env
conda activate ethograph

# Ensure clean working tree first (dirty tree = local version suffix = PyPI rejects)
git status

# Tag the release
git tag v0.1.3

# Build
python -m build

# Upload (uses ~/.pypirc automatically)
python -m twine upload dist/ethograph-0.1.3*
```

## Common pitfalls

- **400 Bad Request: local version** — version like `0.1.3+g6b8d9d2` has a local suffix PyPI rejects. Cause: uncommitted changes when building. Fix: commit everything, retag, rebuild.
- **400 Bad Request: direct dependency** — PyPI rejects `pkg @ git+https://...` and `pkg @ https://...` deps in `pyproject.toml`. Remove them; document manual install steps in the README instead.
- **Locked temp dir on Windows** — build leaves a temp dir that Windows locks. Fix before rebuilding: `Remove-Item -Recurse -Force ethograph-<version>`
- **403 Forbidden** — wrong or truncated API token, or token scoped to wrong project. Use an account-wide token for the first upload.

## Checking the workflow

https://github.com/Akseli-Ilmanen/ethograph/actions
