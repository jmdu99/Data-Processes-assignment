#!/bin/bash

function check_packages {
REQUIRED_PKG=$1
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep "install ok installed")
echo Checking for $REQUIRED_PKG: $PKG_OK
if [ "" = "$PKG_OK" ]; then
  echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."
  sudo apt-get --yes install $REQUIRED_PKG 
fi
}

function check_modules {
REQUIRED_MODULE=$1
MODULE_OK=$(pip3 list --format=columns | grep $REQUIRED_MODULE)
if [ -z "$MODULE_OK" ]
then
      pip3 install $REQUIRED_MODULE
else
      echo "$REQUIRED_MODULE is already installed"
fi
}

check_packages "python3"
check_packages "python3-pip"
check_modules "scikit-learn"

cd "src"
echo "Loading prototype..."
python3 app.py
