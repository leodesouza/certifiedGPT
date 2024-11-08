#!/bin/bash

target_path="/home/leonardosouza/projects/datasets/vqav2/images/"
cd "$target_path" || exit

# Download the train2014.zip
mkdir -p train2014
cd train2014
wget http://images.cocodataset.org/zips/train2014.zip
cd ..


# Download the val2014.zip
mkdir -p val2014
cd val2014
wget http://images.cocodataset.org/zips/val2014.zip
cd ..

# Download the test2014.zip
mkdir -p test2014
cd test2014
wget http://images.cocodataset.org/zips/test2015.zip
cd ..






