#!/bin/bash

# this opens the setup.py file and increments the version number by one in order to publish a new
# version of jeffutils

# Read the current version from setup.py
current_version=$(grep -oP "(?<=version=')[0-9]+\.[0-9]+\.[0-9]+(?=')" setup.py)

# Split the version into major, minor, and patch
IFS='.' read -r major minor patch <<< "$current_version"

# Increment the version
if [ "$patch" -lt 9 ]; then
  patch=$((patch + 1))
else
  patch=0
  if [ "$minor" -lt 9 ]; then
    minor=$((minor + 1))
  else
    minor=0
    major=$((major + 1))
  fi
fi

# Form the new version
new_version="$major.$minor.$patch"

# Update setup.py with the new version
sed -i "s/version='$current_version'/version='$new_version'/" setup.py

echo "Updated version: $current_version -> $new_version"
