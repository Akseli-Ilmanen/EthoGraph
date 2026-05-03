# Publishing to PyPI

Publishing is automated via GitHub Actions (`.github/workflows/test_and_deploy.yml`).
It triggers on **git tags**, not on every push to main.

## One-time setup

Add your PyPI API token as a GitHub secret:
1. Go to repo → Settings → Secrets and variables → Actions
2. New repository secret: name `TWINE_API_KEY`, value = your PyPI token (starts with `pypi-`)

## Release steps

1. Bump the version in `pyproject.toml` (or wherever it's set)
2. Commit and push to main
3. Tag and push the tag:
   ```
   git tag v0.1.3
   git push origin v0.1.3
   ```
4. GitHub Actions runs: lint → tests → build → upload to PyPI automatically

## Manual upload (fallback)

If you need to publish without a tag (e.g. for a dev release):

```powershell
# Activate env first
conda activate ethograph

# Build
python -m build

# Upload (username: __token__, password: your PyPI key)
python -m twine upload dist/<package-version>*
```

## Checking the workflow

https://github.com/Akseli-Ilmanen/ethograph/actions
