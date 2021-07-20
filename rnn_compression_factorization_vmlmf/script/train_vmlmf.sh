cd ./src/
echo "Start training compressed model..."
python ./train_test/main_total.py --model vmmodel_final --layer_sizes 180 --wRank 8 --uRanks 4 -train --seed 3 --gpu_id 2
echo "Test Baseline model.."
python ./train_test/main.py --model mylstm --layer_sizes 180 --seed 3
