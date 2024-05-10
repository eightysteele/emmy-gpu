#!/usr/bin/env bash

set -e

print_var() {
    echo "$1: ${!1}"
}

apt-get update
apt-get install -y \
        rlwrap

curl -L -O https://github.com/clojure/brew-install/releases/latest/download/linux-install.sh

chmod +x linux-install.sh

./linux-install.sh

rm linux-install.sh

