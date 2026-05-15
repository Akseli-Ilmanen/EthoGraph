# Contributing

Contributions are welcome! Whether it's bug reports, feature requests, or pull
requests — all help is appreciated.

## Reporting issues

Open an issue on the
[GitHub repository](https://github.com/Akseli-Ilmanen/ethograph/issues) with a
clear description of the problem and steps to reproduce it. Please:

1) In `Navigation & Help`, click on `Print for debugging`. Share this message along with your error.
2) If you have data loading problems, send some sample data to akseli.ilmanen@gmail.com, so I can test it myself.


## Development installation

To install ethograph in editable mode with all optional dependencies and
development tools:

```bash
git clone https://github.com/Akseli-Ilmanen/ethograph
cd ethograph
uv pip install -e ".[gui,audio,dandi,dev,docs]"
```

See {doc}`../getting_started/installation` for details on setting up a virtual
environment and installing uv.

## Setting up pre-commit hooks

The project uses [pre-commit](https://pre-commit.com/) to run ruff, mypy,
check-manifest, and codespell automatically before every commit. Install the
hooks once after cloning:

```bash
pre-commit install
```

After this, every `git commit` runs the checks locally — the same checks that
run in CI. If any check fails, the commit is blocked and the errors are shown
in the terminal.

To run the checks manually without committing:

```bash
pre-commit run --all-files
```

To auto-fix ruff issues:

```bash
ruff check --fix ethograph/
ruff format ethograph/
```
