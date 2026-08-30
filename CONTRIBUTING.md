# Contributing

Thank you for improving this project.

## Before you start

- Search existing issues and open a focused issue before a large or breaking change.
- Confirm that test data, media, model weights, and documents can legally be redistributed.
- Do not include credentials, personal data, confidential captures, or proprietary project information.
- Keep safety-critical assumptions and limitations visible.

## Development workflow

1. Create a short-lived branch.
2. Make one coherent change.
3. Add or update tests where executable behaviour changes.
4. Update the README and examples when interfaces or assumptions change.
5. Run the repository's documented verification commands.
6. Describe the problem, approach, validation, and remaining limitations in the pull request.

Use clear commit messages and avoid unrelated formatting churn. Maintainers may ask for a smaller change when review or validation would otherwise be difficult.

## Engineering quality

New calculations must identify units and assumptions. Hardware changes need safe test conditions. Data or ML changes need provenance, a reproducible split, metrics, and known limitations. User-facing changes should remain accessible and avoid hiding important warnings.
