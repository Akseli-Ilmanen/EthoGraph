# Publishing to PyPI

Publishing is automated via GitHub Actions (`.github/workflows/test_and_deploy.yml`).
It triggers on **git tags**, not on every push to main.

## One-time setup

API token in `C:\Users\aksel\.pypirc`:

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

