# Contributing to A176LAB RAG System V2

Thank you for your interest in contributing. This document explains how to get started.

## Development setup

```bash
git clone https://github.com/MuhammadAbbasi/RAG-for-renewable-energy.git
cd RAG-for-renewable-energy
pip install -r requirements.txt
```

Start the stack locally with Docker:
```bash
docker compose up -d
```

## Branching strategy

| Branch pattern | Purpose |
|---|---|
| `main` | Stable, production-ready code |
| `feature/<name>` | New features |
| `fix/<name>` | Bug fixes |
| `docs/<name>` | Documentation only |

Always branch off `main` and open a pull request back into `main`.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add /review command for proposal analysis
fix: reduce embedder batch size to prevent LLM timeout
docs: update README hardware requirements
refactor: split lifecycle.py into separate modules
```

## Pull requests

- Keep PRs focused — one feature or fix per PR
- Include a clear description of what changed and why
- Make sure `python -m py_compile` passes on all changed `.py` files
- Reference any related issues with `Closes #N`

## Reporting bugs

Use the **Bug report** issue template. Include:
- Steps to reproduce
- Expected vs actual behaviour
- Docker / Python / OS version
- Relevant log output from `logs/rag_*.log`

## Feature requests

Use the **Feature request** issue template. Describe the use case before the solution.

