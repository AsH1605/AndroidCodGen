#!/bin/bash

# Ensure correct usage
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <new-name> <new-email>"
    exit 1
fi

NEW_NAME="$1"
NEW_EMAIL="$2"

echo "Rewriting Git history with new author name: '$NEW_NAME' and email: '$NEW_EMAIL'..."

# Create a temporary file for commit mapping
COMMIT_MAP_FILE=$(mktemp)
COMMITS_FILE=$(mktemp)

# Get all commits in chronological order (oldest first)
git rev-list --reverse --all > "$COMMITS_FILE"

# Variable to track the latest commit in the rewritten history
NEW_HEAD=""

while read -r commit; do
    # Extract commit details
    tree=$(git rev-parse "$commit^{tree}")
    parent=$(git rev-parse "$commit^" 2>/dev/null || echo "")
    commit_msg=$(git log --format=%B -n 1 "$commit")
    commit_date=$(git log --format=%aI -n 1 "$commit")

    # Check if this commit has a parent (not the root commit)
    if [[ -z "$parent" ]]; then
        # Root commit (no parent)
        new_commit=$(echo "$commit_msg" | GIT_COMMITTER_NAME="$NEW_NAME" GIT_COMMITTER_EMAIL="$NEW_EMAIL" \
                     GIT_COMMITTER_DATE="$commit_date" git commit-tree "$tree" --author="$NEW_NAME <$NEW_EMAIL>")
    else
        # Fetch mapped parent commit
        mapped_parent=$(grep "^$parent " "$COMMIT_MAP_FILE" | awk '{print $2}')
        new_commit=$(echo "$commit_msg" | GIT_COMMITTER_NAME="$NEW_NAME" GIT_COMMITTER_EMAIL="$NEW_EMAIL" \
                     GIT_COMMITTER_DATE="$commit_date" git commit-tree "$tree" -p "$mapped_parent" --author="$NEW_NAME <$NEW_EMAIL>")
    fi

    # Save the commit mapping
    echo "$commit $new_commit" >> "$COMMIT_MAP_FILE"

    # Update HEAD reference to the latest commit
    NEW_HEAD="$new_commit"
done < "$COMMITS_FILE"

# Confirm before resetting
read -p "Are you sure you want to reset history? This is irreversible! (yes/no): " confirm
if [[ "$confirm" == "yes" ]]; then
    git reset --hard "$NEW_HEAD"
    echo "Git history successfully rewritten!"
    echo "To apply these changes remotely, run: git push --force --all"
else
    echo "Operation canceled."
fi

# Cleanup temporary files
rm -f "$COMMITS_FILE" "$COMMIT_MAP_FILE"
