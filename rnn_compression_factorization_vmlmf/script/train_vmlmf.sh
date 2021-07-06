cd ./src/
echo "Start training vanilla_lstm"
python ./train_test/main.py --model mylstm --layer_sizes 180 -train
echo "Start training..."
python ./train_test/main.py --model vmmodel --layer_sizes 180 --wRank 8 --uRanks 6 -train --seed 4 
echo "Test baseline model.."
python ./train_test/main.py --model mylstm --layer_sizes 180
echo "Test compressed model.."
python ./train_test/main.py --model vmmodel --layer_sizes 180 --wRank 8 --uRanks 6 --seed 4
