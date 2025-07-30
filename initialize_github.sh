#!/bin/bash
# Initialize a git repository, add all files, make the first commit, and provide push instructions

git init

git add .

git commit -m "Initial commit: Python4Physics 2025 final submission"

echo "\nRepository initialized and first commit made."
echo "To push to GitHub, run:"
echo "  git remote add origin <your-github-repo-url>"
echo "  git branch -M main"
echo "  git push -u origin main"
