#!/bin/bash
set -euo pipefail

if [[ $(id -u) -eq 0 ]]; then
	>&2 echo "Please don't run this as root!"
	exit 1
fi

srcdir=$(dirname "$(dirname "$(realpath "$0")")")
mkdir -vp ~/.local/opt
cp -v "$srcdir" ~/.local/opt/pumice
uv sync
cd ~/.local/opt/pumice
mkdir -vp ~/.local/bin
chmod +x ~/.local/opt/pumice/scripts/pumice
ln -vsf ~/.local/opt/pumice/scripts/pumice ~/.local/bin
