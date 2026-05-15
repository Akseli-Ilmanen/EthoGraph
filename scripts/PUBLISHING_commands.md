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

## Steps before every push

1. Run the full check suite (same as CI) and let it auto-fix what it can:
   ```powershell
   pre-commit run --all-files
   ```
   If it reports failures, run it **a second time** — pre-commit fixes files in place on the first pass, and the second run confirms everything is clean.

2. Commit all changes (including any files pre-commit just fixed):
   ```powershell
   git add -A
   git commit -m "fix: linting"
   ```

3. Push to main:
   ```powershell
   git push origin main
   ```

## Release steps

1. Ensure working tree is clean (`git status` shows nothing modified)
2. Commit and push all changes to main (see steps above)
3. Check what version you are on and pick the next one:
   ```powershell
   git describe --tags --long --match "v*"
   # e.g. v0.1.8-5-gabcd123 → on v0.1.8, next release is v0.1.9
   ```
4. Tag and push the tag:
   ```powershell
   git tag v0.1.9
   git push origin v0.1.9
   ```
5. GitHub Actions runs: lint → tests → build → upload to PyPI automatically

## Check installed version

```powershell
python -c "import ethograph; print(ethograph.__version__)"
```

If it shows `0.1.dev...`, reinstall to pick up the latest tag:

```powershell
uv pip install -e ".[gui,audio,dandi,dev,docs]"
```

## Monitor the release

| What | URL |
|------|-----|
| CI progress (Actions tab) | https://github.com/Akseli-Ilmanen/ethograph/actions |
| PyPI package page | https://pypi.org/project/ethograph/ |

The workflow typically takes 3–5 minutes. The new version appears on PyPI within a minute of the workflow completing.
