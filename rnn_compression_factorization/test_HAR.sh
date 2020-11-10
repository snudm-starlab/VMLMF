#!/bin/sh

mkdir "./weights"

if [[ "$1" == "HAR_deepconv" ]]; then
  echo "******************************************************************"
  echo "Test for Oppurtunity HAR dataset"
  echo "******************************************************************"
  echo "baseline Deep Convolutional LSTM model test result"
  echo "******************************************************************"

  python HAR_test_for_deepconv_lowrank.py --layer_sizes 128 128 --max-steps 20000 --model myLSTM --gpu 0 --lr 0.0002 --log_epoch -1

  echo "******************************************************************"
  echo "Lowrank compression result on DeepConvLSTM model, compression rate=7"
  echo "******************************************************************"

  python HAR_test_for_deepconv_lowrank.py --layer_sizes 128 128 --uRank 19 --wRank 19 --max-steps 20000 --model myLSTM --gpu 0 --lr 0.0002 --log_epoch -1

  echo "*******************************************************************"
  echo "Proposed group-lowrank compression result on DeepConvLSTM model, compression rate=7"
  echo "******************************************************************"

  python HAR_test_for_deepconv_group_LowRank.py --layer_sizes 128 128 --uRanks 28 10 --wRank 19 --max-steps 20000 --model myLSTM_group --group 2 --gpu 0 --lr 0.0002 --log_epoch -1

elif [[ "$1" == "HAR" ]]; then
  echo "******************************************************************"
  echo "Test for Oppurtunity HAR dataset"
  echo "******************************************************************"
  echo "baseline LSTM model test result"
  echo "******************************************************************"

  python HAR_test_for_LowRank.py --layer_sizes 64 64 --max-steps 10000 --model myLSTM --gpu 0 --lr 0.002 --log_epoch -1

  echo "******************************************************************"
  echo "Lowrank compression result on LSTM model, compression rate=7"
  echo "******************************************************************"

  python HAR_test_for_LowRank.py --layer_sizes 64 64 --uRank 9 --wRank 9 --max-steps 10000 --model myLSTM --gpu 0 --lr 0.002 --log_epoch -1

  echo "*******************************************************************"
  echo "Proposed group-lowrank compression result on LSTM model, compression rate=7"
  echo "******************************************************************"

  python HAR_test_for_group_LowRank.py --layer_sizes 64 64 --uRanks 12 6 --wRank 9 --max-steps 10000 --model myLSTM_group --group 2 --gpu 0 --lr 0.002 --log_epoch -1

elif [[ "$1" == "UCI" ]]; then
  echo "******************************************************************"
  echo "Test for UCI_HAR dataset"
  echo "******************************************************************"
  echo "baseline LSTM model test result"
  echo "******************************************************************"

  python UCI_test_for_LowRank.py --layer_sizes 64 64 --max-steps 10000 --model myLSTM --gpu 0 --lr 0.002 --log_epoch -1

  echo "******************************************************************"
  echo "Lowrank compression result on LSTM model, compression rate=7"
  echo "******************************************************************"

  python UCI_test_for_LowRank.py  --layer_sizes 64 64 --uRank 9 --wRank 9 --max-steps 10000 --model myLSTM --gpu 0 --lr 0.002 --log_epoch -1

  echo "*******************************************************************"
  echo "Proposed group-lowrank compression result on LSTM model, compression rate=7"
  echo "******************************************************************"

  python UCI_test_for_group_LowRank.py --layer_sizes 64 64 --uRanks 12 6 --wRank 9 --max-steps 10000 --model myLSTM_group --group 2 --gpu 0 --lr 0.002 --log_epoch -1

else
  echo "specify Dataset and model"

fi