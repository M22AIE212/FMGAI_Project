#!/bin/bash

# Download files
wget https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar
wget https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar

# Extract files
mkdir -p abo-images-small
tar -xf abo-images-small.tar -C abo-images-small

mkdir -p abo-listings
tar -xf abo-listings.tar -C abo-listings