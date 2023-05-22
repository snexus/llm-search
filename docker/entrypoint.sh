#!/usr/bin/env bash
set -e

if [[ -z "$HOST_UID" ]]; then
    echo "ERROR: please set HOST_UID" >&2
    exit 1
fi
if [[ -z "$HOST_GID" ]]; then
    echo "ERROR: please set HOST_GID" >&2
    exit 1
fi

# Use this code if you want to create a new user account:
addgroup --gid "$HOST_GID" dockergroup
adduser --uid "$HOST_UID" --gid "$HOST_GID" --gecos "" --disabled-password app

# -OR-
# Use this code if you want to modify an existing user account:
#groupmod --gid "$HOST_GID" app
#usermod --uid "$HOST_UID" app

# Drop privileges and execute next container command, or 'bash' if not specified.
if [[ $# -gt 0 ]]; then
    exec sudo -H -u  app --  "$@"
else
    exec sudo -H -u app -- bash
fi
