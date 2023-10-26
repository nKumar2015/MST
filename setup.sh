#!/bin/bash

mkdir data

cd data

echo "Downloading Metadata (358 MB)"
curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip

echo "Downloading Dataset (7.68 GB)"
curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip

echo "Unzipping"

echo "Unzipping Metadata"
unzip  fma_metadata.zip

echo "Unzipping Dataset"
unzip  fma_small.zip
