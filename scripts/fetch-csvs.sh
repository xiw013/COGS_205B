#!/bin/bash

DOWNLOAD_URL="data.zip https://raw.githubusercontent.com/joachimvandekerckhove/cogs205b-s26/main/modules/02-version-control/files/data.zip"
cd /workspace/repo/scripts
wget -O data.zip ${DOWNLOAD_URL}
unzip -q data.zip -d tempdir
mkdir -p /workspace/repo/data/$(date +%F)
find tempdir -maxdepth 1 -type f -name "*.csv" -exec mv {} /workspace/repo/data/$(date +%F)/ \;
rm -rf /workspace/repo/scripts/tempdir
rm /workspace/repo/scripts/data.zip

cd /workspace/repo
git add /workspace/repo/data/ scripts/fetch-csvs.sh
git commit -m "Add data files for $(date +%F) and fetch script."

git remote add origin git@github.com:xiw013/COGS_205B.git
git push origin master