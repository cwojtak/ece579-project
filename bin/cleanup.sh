#!/bin/bash

processed_dir="data/processed/*"
train_split_dir="data/split/train/*"
test_split_dir="data/split/test/*"

cleanup_dir() {
    dir=$1
    echo "Removing $dir"
    rm -rf $dir
}

cleanup_dir "$processed_dir"
cleanup_dir "$train_split_dir"
cleanup_dir "$test_split_dir"

echo "Processed and split data files removed."
