# Repository Guidelines

## Project Structure & Module Organization

This repository is currently an empty project scaffold. Establish a predictable layout as implementation begins:

- `src/` for application or library source code.
- `tests/` for automated tests, mirroring the relevant paths in `src/`.
- `assets/` for checked-in static resources such as fixtures or images.
- `docs/` for design notes and user-facing documentation.

Keep generated output, dependency directories, credentials, and local environment files out of version control. Add new top-level directories only when their purpose is clear and documented.

## Build, Test, and Development Commands

No build system or package manifest has been committed yet. When adding one, document the canonical commands here and in the project README. At minimum, provide commands for:

- installing dependencies (for example, `npm install`),
- running the application locally,
- running the full test suite, and
- formatting and linting changed code.

Use a single, repeatable command for each task so contributors and CI execute the same checks.

## Coding Style & Naming Conventions

Follow the formatter and linter configured for the chosen language; do not hand-format around their output. Use descriptive, lowercase directory and file names appropriate to the ecosystem (for example, `data_loader.py` or `data-loader.ts`). Keep modules focused, avoid unexplained abbreviations, and place configuration defaults in tracked example files rather than personal environment files.

## Testing Guidelines

Add tests with every behavior change and keep them deterministic and isolated from external services. Name test files according to the selected framework (such as `*.test.ts` or `test_*.py`) and use test descriptions that state expected behavior. Before opening a pull request, run the full test, lint, and formatting checks locally.

## Commit & Pull Request Guidelines

There is no commit history yet, so no repository-specific convention exists. Use concise imperative commit subjects, such as `Add tokenizer configuration`. Keep commits focused and avoid mixing refactors with behavior changes. Pull requests should explain the change, note validation performed, link relevant issues, and include screenshots or sample output when user-visible behavior changes.

## Security & Configuration

Never commit secrets, tokens, private datasets, or machine-specific configuration. Provide a sanitized `.env.example` when environment variables are required, and document every required setting.
