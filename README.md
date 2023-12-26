# Causal Representation Learning for Out-of-Distribution Recommendation
This is the pytorch implementation of our paper at WWW 2022:
> Causal Representation Learning for Out-of-Distribution Recommendation
>
> Wenjie Wang, Xinyu Lin, Fuli Feng, Xiangnan He, Min Lin, Tat-Seng Chua

## Environment
- Anaconda 3
- python 3.7.3
- pytorch 1.4.0
- numpy 1.16.4

## Usage

### Data
The experimental data are in './data' folder, including Synthetic Data, Meituan and Yelp. Due to the large size, 'item_feature.npy' of Yelp is uploaded to [Google drive](https://drive.google.com/drive/folders/1nKk15UlYzGVKCo5yMFVmW4yewbwid0dH?usp=sharing).

### Training
```
python main.py --model_name=$1 --dataset=$2 --mlp_dims=$3 --mlp_p1_1_dims=$4 --mlp_p1_2_dims=$5 --mlp_p2_dims=$6 --mlp_p3_dims=$7 --lr=$8 --wd=$9 --batch_size=$10 --epochs=$11 --total_anneal_steps=$12 --anneal_cap=$13 --CI=$14 --dropout=$15 --Z1_hidden_size=$16 --E2_hidden_size=$17 --Z2_hidden_size=$18 --bn=$19 --sample_freq=$20 --regs=$21 --act_function=$22 --log_name=$23 --gpu=$24 --cuda
```
or use run.sh
```
sh run.sh model_name dataset mlp_dims mlp_p1_1_dims mlp_p1_2_dims mlp_p2_dims mlp_p3_dims lr wd batch_size epochs total_anneal_steps anneal_cap CI dropout Z1_hidden_size E2_hidden_size Z2_hidden_size bn sample_freq regs act_function log_name gpu_id
```
- The log file will be in the './code/log/' folder. 
- The explanation of hyper-parameters can be found in './code/main.py'. 
- The default hyper-parameter settings are detailed in './code/hyper-parameters.txt'.

### Inference
Get the results of COR over iid and ood data where only user features are drifted by running inference.py:

```
python inference.py --dataset=$1 --ckpt=$2 --cuda
```

### Fine-tuneing
```
python main.py --model_name=$1 --dataset=$2 --X=$3 --lr=$4 --wd=$5 --batch_size=$6 --epochs=$7 --total_anneal_steps=$8 --anneal_cap=$9 --CI=$10 --dropout=$11 --bn=$12 --sample_freq=$13 --regs=$14 --log_name=$15 --ckpt=$16 --gpu=$17 --ood_finetune --cuda
```
or use finetune.sh
```
sh finetune.sh model_name dataset X lr wd batch_size epochs total_anneal_steps anneal_cap CI dropout bn sample_freq regs log_name <pre-trained model directory> gpu_id
```
- The log file will be in the './code/log/finetune/' folder.


### Examples

1. Train COR on iid meituan:

```
cd ./code
sh run.sh COR meituan [3000] [] [1] [] [] 1e-3 0 500 300 0 0.1 1 0.5 500 1000 200 0 1 0 tanh log 0
```

2. Inference on synthetic data:

```
cd ./code
python inference.py --dataset synthetic --ckpt <pre-trained model directory> --cuda
```

3. Fine-tuning COR on ood yelp:
```
cd ./code
sh finetune.sh COR yelp 0 0.0001 0.03 500 0 0.5 1 0.4 0 1 0 log <pre-trained model directory> 0
```
## License

NUS © [NExT++](https://www.nextcenter.org/)