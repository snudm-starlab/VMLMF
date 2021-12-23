#! /bin/sh
cd ./src/
echo "Start training compressed model...mylstm"
python ./train_test/main_total.py --model mylstm --layer_sizes 180 --seed 3 --gpu_id 0

echo "Start training compressed model...vmmodel"
python ./train_test/main_total.py --model vmmodel --layer_sizes 180 --wRank 8 --uRanks 6 --seed 3 --gpu_id 0

echo "Start training compressed model...vmlmf_group2 2/4"
python ./train_test/main_total.py --model vmlmf_group2 --layer_sizes 180 --wRank 8 --uRanks 2 4 --seed 3 --gpu_id 0
