#!/bin/bash

# make and copy data to directory that complies with torchvision.datasets.ImageFolder
# usage: bash utils/copy_csv.sh --ALVR_ROOT /path/to/lig_data 
#
# structure of ALVR_ROOT directory should be like: 
# ALVR_ROOT/
# ├── p1
# │   ├── a/*.csv
# │   ├── e/*.csv
# │   ├── i/*.csv
# │   ├── o/*.csv
# │   └── u/*.csv
# ├── p2
# │   ├── a/*.csv
# │   ├── e/*.csv
# ...
# ...


ALVR_ROOT=`realpath ~/workspace/LIG/rawdata`
valid_spkrs="p1"
train_spkrs="p2 p3 p4 p5"
datadir=
nj=8

. utils/parse_options.sh

for phn in a e i o u; do
    mkdir -p $datadir/train/$phn/
    for spkr in $train_spkrs; do
        ln -s $ALVR_ROOT/$spkr/$phn/*.csv $datadir/train/$phn/
    done

    mkdir -p $datadir/valid/$phn/
    for spkr in $valid_spkrs; do
        ln -s $ALVR_ROOT/$spkr/$phn/*.csv $datadir/valid/$phn/
    done
done

find $datadir -iname "*.csv" > $datadir/csv.flist
total_lines=`wc -l $datadir/csv.flist | cut -d" " -f1`
if [ $nj -eq 1 ]; then
    split_lines=${total_lines}
else
    split_lines=$((total_lines/(nj-1)))
fi
split -l ${split_lines} --numeric-suffixes $datadir/csv.flist $datadir/csv.flist.part

