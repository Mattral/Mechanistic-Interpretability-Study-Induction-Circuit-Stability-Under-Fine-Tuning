# Contributing

## Before you start

Read `decisions/DECISION_LOG.md` and the project specification (owner-held
AGENT_INSTRUCTIONS) to understand all fixed decisions. Never alter the
hypothesis (Section 1.3), the induction score formula (Section 4.2), or the
patching protocol (Section 4.3) without a logged decision.

## Workflow

1. Create a branch: `git checkout -b <type>/<scope>/<short-description>`
2. Write code conforming to Section 3 standards.
3. Run the pre-commit checks:
   ```bash
   black src/ tests/
   ruff check src/ tests/
   mypy src/
   pytest tests/ --tb=short -q
   ```
4. Commit using the format in README.md § Commit Standards.
5. Open a PR; self-review against the spec before requesting review.
6. CI must be green before merge. No exceptions.

## Adding a decision

Any non-obvious choice → new entry in `decisions/DECISION_LOG.md` using the
template in Section 11 of AGENT_INSTRUCTIONS.

## Updating dependencies

1. Test the full pipeline locally.
2. Update both `requirements.txt` and `environment.yml`.
3. Log the reason in `decisions/DECISION_LOG.md`.
4. Tag the commit: `deps: update <package> to <version>`.
