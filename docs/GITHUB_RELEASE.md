# Publish this release on GitHub

## Repository setup

Suggested repository name: `RMD-ES-research-artifacts`. Create it with **Public** visibility if the authors and rights holders have approved public release. TVT's official author instructions specify single-anonymous review, so an anonymous repository is not required for this journal. See [`TVT_REPOSITORY_NOTE.md`](TVT_REPOSITORY_NOTE.md).

Use this prepared folder as the repository root. Do not publish the parent research workspace: it contains unrelated material, logs, editor state, and other assets excluded from this release.

## Push with Git

Create an empty public repository on GitHub without initializing it with a README, license, or `.gitignore`, then run these commands from this folder:

```bash
git init -b main
git add .
git status --short
git commit -m "Publish RMD-ES research code and data"
git remote add origin https://github.com/OWNER/RMD-ES-research-artifacts.git
git push -u origin main
```

Replace `OWNER` with the GitHub account or organization that will own the repository. GitHub Desktop can also publish this folder as a new public repository. Do not put access tokens or passwords in a remote URL or repository file.

## Before adding the article link

Review the staged files and repository visibility with the coauthors and relevant institutions. The repository has no code/data license yet; resolve that separately rather than adding a default license automatically. Add the GitHub URL to the article only after the public repository exists and the contents are final. If the journal later assigns a DOI through a repository/archive service, cite that persistent record as well.
