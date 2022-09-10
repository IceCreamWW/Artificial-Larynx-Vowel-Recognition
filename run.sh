#!/bin/bash
# bash run.sh --stage 2 --stop-stage 2 --n_seeds 10 --model resnet18
set -e
set -u
set -o pipefail

ALVR_ROOT=`realpath ~/workspace/LIG/rawdata`
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

# common
stage=1
stop_stage=1
datadir=
python=python3

# feature extraction
fs=48000
fmin=0
fmax=8000
win_length=2048
hop_length=512
n_fft=2048
n_mels=141
img_size=256

# training
n_seeds=20
max_epoch=25
model="resnet18" # resnet18 resnet50 densenet161


. utils/parse_options.sh

if [ -z $datadir ]; then
    datadir=data/fs${fs}_fmin${fmin}_fmax${fmax}_n_mels${n_mels}_win${win_length}_hop${hop_length}_nfft${n_fft}_size${img_size}
fi

if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: create spectrogram data"
    rm -rf $datadir
    utils/copy_csv.sh --ALVR_ROOT ${ALVR_ROOT} --datadir ${datadir}
    pids=""
    for f in $datadir/csv.flist.part*; do
        ${python} utils/csv2img.py \
            --flist $f \
            --fs ${fs} --fmin ${fmin} --fmax ${fmax}  --n-mels ${n_mels} \
            --win-length ${win_length} --hop-length ${hop_length} --n-fft ${n_fft} \
            --size ${img_size} &
        pids="$pids $!"
    done
	for pid in $pids; do
		wait $pid || exit 1
	done
    rm $datadir/csv.flist.part*
fi

if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: train ${model} model for ${n_seeds} times"
    for i in `seq 1 ${n_seeds}`; do
        seed=$RANDOM
        expdir=exp/${model}/$(date '+%Y%m%d%H%M%S')_seed_${seed}
        mkdir -p $expdir
        echo "training with seed = $seed ..., log in $expdir/train.log"
        ${python} train.py --seed $seed --model $model --expdir $expdir --data-root $datadir --max-epoch ${max_epoch} > $expdir/train.log
    done
fi

if [ $stage -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: collecting results ..."
    rm -f exp/$model/best.scp exp/$model/last.scp
    for d in exp/$model/*; do 
        if ! [ -f $d/train.log  ] || ! grep -q -i "best valid" $d/train.log; then 
            # echo $d;
            continue
        fi
        grep -i -E "(best valid)|(seed =)" $d/train.log | cut -d" " -f3- | tr '\n' ' ' | awk '{print $3,$NF}' >> exp/$model/best.scp
        grep -i -E "(last valid)|(seed =)" $d/train.log | cut -d" " -f3- | tr '\n' ' ' | awk '{print $3,$NF}' >> exp/$model/last.scp
    done
    ave_best_acc=`awk '{sum+=$2} END{print sum/NR}' exp/$model/best.scp`
    ave_last_acc=`awk '{sum+=$2} END{print sum/NR}' exp/$model/last.scp`
    max_best_acc=`sort -k2 -nr exp/$model/best.scp | head -n 1 | awk '{print $2}'`
    max_last_acc=`sort -k2 -nr exp/$model/last.scp | head -n 1 | awk '{print $2}'`
    echo "Averaged best acc: ${ave_best_acc}    Maximum best acc: ${max_best_acc}"
    echo "Averaged last acc: ${ave_last_acc}    Maximum last acc: ${max_last_acc}"
fi

if [ $stage -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: test with the best checkpoint"
    seed=`sort -k2 -nr exp/$model/best.scp | head -n 1 | awk '{print $1}'`
    best_acc_ckpt_path=`find exp/$model -name "*seed_$seed"`/valid.acc.best.pth
    echo "Testing with best acc checkpoint path: ${best_acc_ckpt_path}"
    ${python} test.py --model $model --ckpt-path ${best_acc_ckpt_path} --data-root $datadir
fi

if [ $stage -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Stage 5: clean model directory"
    for d in exp/$model/*; do 
        if ! [ -f $d/train.log  ] || ! grep -q -i "best valid" $d/train.log; then 
            echo "removing $d"
            rm -rf $d
        fi
    done
    echo "cleaned model directory exp/$model"
fi

