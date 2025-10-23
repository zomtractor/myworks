git pull
cd work2_1
python download.py
nohup tensorboard --logdir=checkpoints/DeFlare/train_logs > tensorboard.log &
python train_online.py
