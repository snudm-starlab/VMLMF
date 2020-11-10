#!/bin/sh
python HAR_test_for_deepconv_group_LowRank2.py  --layer_sizes 128 128 --uRanks 28 10 --wRank 19 --max-steps 1000 --model myLSTM_group --group 2 --gpu 1 --lr 0.0002 --log_epoch -1