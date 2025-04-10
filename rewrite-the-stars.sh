#!/bin/bash

# Ensure correct usage
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <new-name> <new-email>"
    exit 1
fi

NEW_NAME="$1"
NEW_EMAIL="$2"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Error: Not inside a Git repository."
    exit 1
fi

if ! git rev-list --all | grep -q .; then
    echo "Error: No commits found in the repository."
    exit 1
fi

echo "Rewriting Git history with new author name: '$NEW_NAME' and email: '$NEW_EMAIL'..."

# Create temporary files
COMMIT_MAP_FILE=$(mktemp)
COMMITS_FILE=$(mktemp)

# Get all commits in chronological order
git rev-list --reverse --all > "$COMMITS_FILE"

NEW_HEAD=""

while read -r commit; do
    tree=$(git rev-parse "$commit^{tree}")

    # Ensure the parent commit exists
    if git cat-file -e "$commit^" 2>/dev/null; then
        parent=$(git rev-parse "$commit^")
    else
        parent=""
    fi

    commit_msg=$(git log --format=%B -n 1 "$commit")
    commit_date=$(git log --format=%aI -n 1 "$commit")

    if [[ -n "$parent" ]]; then
        mapped_parent=$(grep "^$parent " "$COMMIT_MAP_FILE" | awk '{print $2}')
        if [[ -z "$mapped_parent" ]]; then
            echo "Error: No mapped parent commit found for $commit."
            exit 1
        fi
    else
        mapped_parent=""
    fi

    # Create the new commit
    if [[ -z "$mapped_parent" ]]; then
        new_commit=$(echo "$commit_msg" | GIT_COMMITTER_NAME="$NEW_NAME" GIT_COMMITTER_EMAIL="$NEW_EMAIL" \
                     GIT_COMMITTER_DATE="$commit_date" git commit-tree "$tree" --author="$NEW_NAME <$NEW_EMAIL>")
    else
        new_commit=$(echo "$commit_msg" | GIT_COMMITTER_NAME="$NEW_NAME" GIT_COMMITTER_EMAIL="$NEW_EMAIL" \
                     GIT_COMMITTER_DATE="$commit_date" git commit-tree "$tree" -p "$mapped_parent" --author="$NEW_NAME <$NEW_EMAIL>")
    fi

    # Save the mapping
    echo "$commit $new_commit" >> "$COMMIT_MAP_FILE"
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

# Cleanup
rm -f "$COMMITS_FILE" "$COMMIT_MAP_FILE"
