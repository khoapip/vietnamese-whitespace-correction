#!/bin/bash

echo "Downloading Data"
#gsutil cp -r gs://vietai_public/viT5/data/wikilingua .
#gsutil cp -r gs://vietai_public/viT5/data/vietnews .

# Loop through all files in the current directory
for file in wikilingua/*; do
  python generate_dataset.py --filepath $file
done




