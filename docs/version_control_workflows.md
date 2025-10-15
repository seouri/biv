# Version Control and Development Workflows

**Last Updated**: October 15, 2025

This document outlines Git workflows, branching strategies, and development processes that integrate with Test-Driven Development (TDD) practices.

## Git Branching Strategy

### Feature Branches

- **When**: Start of new features (after Step 1: Understand Requirements).
- **Command**: `git checkout -b feature/[descriptive-name]`.
- **Example**: `git checkout -b feature/cart-total-calculation`.
- **Human Confirmation**: "Human: Confirm feature branch `feature/cart-total-calculation` created?"

### Release Branches

- **When**: Preparing for releases or major versions.
- **Command**: `git checkout -b release/v1.0.0`.
- **Merges**: Bug fixes merged back to develop/main; hotfixes to both release and develop branches.

### Hotfix Branches

- **When**: Critical production bug fixes.
- **Command**: `git checkout -b hotfix/critical-bug-fix`.
- **Merges**: To both main and develop branches.

## Commit Workflow

Integrate commits at the end of each TDD cycle for clean, atomic history.

### Making Commits

**After each Red-Green-Refactor cycle**:

1. **Run Quality Checks**:
   ```bash
   uv run ruff check --fix .
   uv run ruff format .
   uv run pytest --cov --cov-fail-under=85
   uv run mypy src tests
   ```

2. **Stage Changes**:
   ```bash
   git add .
   ```

3. **Strictly Prevent Bypassing Hooks**: Never use `--no-verify` or skip quality checks.

4. **Commit with Conventional Format**:
   ```bash
   git commit -m "feat: handle empty cart total"
   ```

5. **Push Immediately**:
   ```bash
   git push
   ```

**Human Confirmation**: "Human: The test for [behavior] is now passing. Confirm to commit?"

### Conventional Commits Format

- `feat:`: New features
- `fix:`: Bug fixes
- `docs:`: Documentation changes
- `style:`: Code style/formatting (no logic change)
- `refactor:`: Code changes that neither fix bugs nor add features
- `perf:`: Performance improvements
- `test:`: Adding or modifying tests
- `chore:`: Maintenance tasks

### Squashing and Merging

**When**: Feature completion.

1. **Create Pull Request**: Use GitHub/GitLab interface.
2. **Squash and Merge**: Keeps main history clean and feature-focused.
3. **Human Confirmation**: "Human: Confirm feature ready for squash merge?"

## Pull Request Process

### Creating PRs

1. **Branch**: Feature branch ready for merge.
2. **Target**: Usually `main` or `develop`.
3. **Description**: Include test coverage, breaking changes, related issues.
4. **Checklist**:
   - [ ] Tests pass on CI
   - [ ] Code coverage >85%
   - [ ] No linting errors
   - [ ] Type checks pass
   - [ ] Documentation updated

### Review Process

1. **Automated Checks**: CI must pass before review.
2. **Code Review**: Focus on logic, tests, documentation.
3. **Merge Strategy**: Squash for feature branches; merge for releases.
4. **Delete Branch**: After merge.

## Conflict Resolution

### During Merges

1. **Pull Latest**: `git pull origin main --rebase`.
2. **Resolve Conflicts**: Edit files, then `git add <resolved-file>`.
3. **Continue Rebase**: `git rebase --continue`.
4. **Force Push**: `git push --force-with-lease` (only if rebasing).

### Best Practices

- **Rebase vs Merge**: Use rebase for local cleanup; merge for published branches.
- **Atomic Commits**: Each commit should represent a single logical change.
- **Clean History**: Use interactive rebase to squash related commits.

## Integration with TDD

### TDD Cycle Integration

- **Feature Branch**: Created after requirements confirmed (Step 1).
- **Commits**: One per Red-Green-Refactor cycle.
- **PR Creation**: When feature meets coverage and quality requirements.
- **Human Oversight**: All major checkpoints require confirmation.

### Example Workflow

```bash
# 1. Start Feature
git checkout -b feature/add-cart-functionality

# 2. TDD Cycle 1
# Write test, fail, implement, refactor
# Run quality checks
uv run pytest --cov --cov-fail-under=85
git add .
git commit -m "feat: implement cart total calculation"

# 3. Human Confirmation
# (Pause for review)

# 4. Continue Cycles
# Repeat for each behavior

# 5. Create PR
git push -u origin feature/add-cart-functionality
# Open PR on GitHub/GitLab

# 6. After approval and merge
git checkout main
git pull
git branch -d feature/add-cart-functionality
```

## Advanced Workflows

### Git Flow Variant

For larger projects:

- `main`: Production releases
- `develop`: Integration branch
- `feature/*`: Feature development
- `release/*`: Release preparation
- `hotfix/*`: Emergency fixes

### GitHub Flow (Simplified)

For smaller teams:

- `main`: Always deployable
- `feature/*`: Short-lived branches
- Direct merges to main via PRs

## Continuous Integration

### Required Checks

All commits must pass:

- **Unit Tests**: `pytest --cov --cov-fail-under=85`
- **Linting**: `ruff check`
- **Formatting**: `ruff format --check`
- **Type Checking**: `mypy src tests`

### CI Configuration

See [Project Setup Guide](project_setup.md#ci/cd-integration) for detailed CI configuration.

## Troubleshooting

### Common Issues

- **Hook Failures**: Run `pre-commit run --all-files` locally to debug.
- **Merge Conflicts**: Use `git mergetool` or resolve manually in editor.
- **Force Push Issues**: Prefer `--force-with-lease` over `--force`.
- **Commit Amend**: Use `git commit --amend` for fixing last commit (avoid on published branches).

### Recovery Commands

```bash
# Reset to safe state
git reset --soft HEAD~1  # Keep changes
git reset --hard HEAD~1  # Discard changes

# Fix published commit
git revert <bad-commit-hash>

# Clean up local branches
git branch -d $(git branch --merged | grep -v master)
