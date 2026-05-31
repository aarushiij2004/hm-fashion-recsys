#!/bin/bash
# Requires: kaggle CLI configured with ~/.kaggle/kaggle.json

set -e
mkdir -p data/raw
echo "Downloading H&M dataset from Kaggle …"
kaggle competitions download -c h-and-m-personalized-fashion-recommendations -p data/raw
cd data/raw && unzip -o h-and-m-personalized-fashion-recommendations.zip
echo "Done. Files in data/raw/"
