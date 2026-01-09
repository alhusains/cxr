# GitHub Repository Setup Guide

This guide walks you through setting up the GitHub repository for submission.

## Prerequisites

1. GitHub account
2. Git configured locally:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 1: Create Private GitHub Repository

1. Go to https://github.com/new
2. Repository settings:
   - **Name**: `chest-xray-classification` (or your preferred name)
   - **Description**: "Production-ready chest X-ray classification system for SickKids ML Specialist technical assessment"
   - **Visibility**: **Private** (required for assessment)
   - **DO NOT** initialize with README (we already have one)
   - **DO NOT** add .gitignore (we already have one)
   - **DO NOT** add license yet

3. Click "Create repository"

## Step 2: Connect Local Repository to GitHub

GitHub will show you commands. Use these:

```bash
cd /home/alhusain/scratch/cxr

# Add GitHub as remote origin
git remote add origin https://github.com/YOUR_USERNAME/chest-xray-classification.git

# Verify remote is added
git remote -v

# Rename branch to main (GitHub standard)
git branch -M main

# First push
git push -u origin main
```

## Step 3: Add Reviewers as Collaborators

1. Go to your repository: `https://github.com/YOUR_USERNAME/chest-xray-classification`
2. Click **Settings** (top right)
3. Click **Collaborators** (left sidebar)
4. Click **Add people**
5. Add reviewers:
   - `I-Akrout`
   - `Bgreer101`
6. Set permission level: **Read** (they only need to view, not edit)

They will receive email invitations.

## Step 4: Verify Repository Access

Check that:
- [ ] Repository is **private**
- [ ] Both reviewers (`I-Akrout` and `Bgreer101`) are added
- [ ] Repository contains all files from local
- [ ] .gitignore is working (no `data/`, `models/`, `venv/`, `mlruns/`)

## Recommended: Add GitHub Authentication

For easier pushing without password every time:

### Option A: Personal Access Token (Recommended)

1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Name: "CXR Project"
4. Expiration: 30 days
5. Scopes: Check `repo` (full control of private repositories)
6. Click "Generate token"
7. **Copy the token immediately** (you won't see it again)

8. Use token as password when pushing:
```bash
git push
# Username: your_github_username
# Password: paste_your_token_here
```

9. Cache credentials (optional):
```bash
git config --global credential.helper cache
# or for permanent storage:
git config --global credential.helper store
```

### Option B: SSH Key (Alternative)

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add to ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub:
# Go to https://github.com/settings/keys
# Click "New SSH key"
# Paste your public key

# Update remote to use SSH
git remote set-url origin git@github.com:YOUR_USERNAME/chest-xray-classification.git
```

## Development Workflow

### Making changes and pushing:

```bash
# Check status
git status

# Add changes
git add <files>
# or add all changes:
git add .

# Commit with descriptive message
git commit -m "Descriptive message about changes"

# Push to GitHub
git push
```

### Suggested commit messages for phases:

- "Add data acquisition and EDA scripts"
- "Implement data preprocessing pipeline"
- "Add model training infrastructure"
- "Complete hyperparameter tuning"
- "Add comprehensive evaluation metrics"
- "Implement Grad-CAM explainability"
- "Add FastAPI deployment system"
- "Update technical report with results"

### Branch strategy (optional):

```bash
# Create feature branch
git checkout -b feature/data-pipeline

# Make changes, commit

# Push feature branch
git push -u origin feature/data-pipeline

# Merge to main when ready
git checkout main
git merge feature/data-pipeline
git push
```

## Before Final Submission (January 16, 2026)

### Pre-submission checklist:

1. **Code quality**
```bash
make format  # Format code
make lint    # Check linting
make test    # Run tests
```

2. **Documentation**
- [ ] README.md is complete
- [ ] TECHNICAL_REPORT.md is filled out
- [ ] All code has docstrings
- [ ] Configuration files documented

3. **Remove sensitive data**
```bash
# Make sure .gitignore is working
git status  # Should NOT show: data/, models/, venv/, .env
```

4. **Final commit and push**
```bash
git add .
git commit -m "Final submission: complete CXR classification system"
git push
```

5. **Tag the submission**
```bash
git tag -a v1.0-submission -m "Submission for SickKids ML Specialist position"
git push origin v1.0-submission
```

6. **Verify reviewers have access**
   - Go to repository Settings > Collaborators
   - Confirm `I-Akrout` and `Bgreer101` are listed

## Repository Best Practices

### What to commit:
- All source code
- Configuration files
- Documentation
- Small example files
- Requirements files
- Notebooks (clear outputs before committing)

### What NOT to commit:
- Large data files (handled by .gitignore)
- Model checkpoints (too large)
- Virtual environments
- MLflow runs (too large)
- API keys or credentials
- `__pycache__` directories

### If you accidentally commit large files:

```bash
# Remove from git but keep locally
git rm --cached <file>
git commit -m "Remove large file from git"
git push
```

## Getting Help

If something goes wrong:

```bash
# Check remote
git remote -v

# Check branch
git branch -a

# View commit history
git log --oneline

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Discard uncommitted changes
git checkout -- <file>
```

## Timeline Reminder

**Deadline: Friday, January 16, 2026 at 11:59 PM**

Commits after this time will not be considered. Plan to finish by January 15th to have buffer time.

## Summary: Quick First Push

```bash
cd /home/alhusain/scratch/cxr

# Create repo on GitHub (private), then:
git remote add origin https://github.com/YOUR_USERNAME/chest-xray-classification.git
git branch -M main
git push -u origin main

# Add collaborators on GitHub web interface:
# I-Akrout and Bgreer101 (Read access)
```

That's it! Your code is now on GitHub and reviewers can access it.
