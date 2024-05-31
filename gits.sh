#!/bin/bash

# this script adds, commits, pulls, and pushes to git with the message passed
# from the command line
message=$1

echo "Adding, committing, pulling, and pushing to git with message: $message"

git add --all;
git commit -m "$message";
git pull;
git push;