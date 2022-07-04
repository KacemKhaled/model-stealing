#!/bin/bash

# CUBS200 takes a very long time to get trained, so we use a pretrained model and fine-tune it
# training with step gives better results

# train victim : --pretrained  -b 256
# Transfer set construction: --batch_size 8
# train victim : --pretrained imagenet -b 64

DATADIR=./data
MODELDIR=./models

dataset=CUBS200  # Caltech256  #CUBS200 # SVHN # CIFAR10 # CIFAR100
victim_arch=resnet34
query_set=ImageNet1kval
adv_arch=resnet34 # resnet34 #vgg16
budgets=50000,45000,40000,35000,30000,25000,20000,15000,10000,5000,1000
budget=50000
suffix='-val'
victim_select=0 # 0: most recent, -1 oldest, -2 the best from all
epochs=200
epochs_knockoff=100

victim_base_dir="$MODELDIR"/victim/"$dataset"-"$victim_arch"
output_model_knockoff_dir="$MODELDIR"/adversary/"$dataset"-"$adv_arch"-random"$suffix"
# Adversarial Training
ratio=2
victim_robust_arch=resnet34
attack=pgd # fgsm # pgd
eps=0,0.01,0.03,.05,.1,.15,.2,.25,.3
eps_adv_robust=0.03,0.05
adv_file="$attack"
### Transfer set construction using as a victim the robust classifier
##
victim_adv_robust_dir="$victim_base_dir"/"$dataset"-"$victim_robust_arch"-"$adv_file"
output_model_knockoff_robust_dir="$output_model_knockoff_dir"-"$adv_file"

echo "$dataset" "$victim_arch" "$adv_arch" "$attack" "$eps"
echo $victim_base_dir
echo $output_model_knockoff_dir
echo $victim_adv_robust_dir
echo $output_model_knockoff_robust_dir

echo "------------------------------------1-----------------------------------------"
echo "------------------------------Victim Training---------------------------------"
echo "------------------------------------------------------------------------------"

python ms/utils/train.py "$dataset" "$DATADIR" -a "$victim_arch" --epochs "$epochs" -o "$victim_base_dir" \
    --scheduler_choice step --batch_size 256 --pretrained

echo "-------------------------------------2----------------------------------------"
echo "--------------------------Transfer set construction---------------------------"
echo "------------------------------------------------------------------------------"

python ms/attacks/extraction/knockoff_nets_transfer.py "$victim_base_dir" \
       --out_dir "$output_model_knockoff_dir" --budget "$budget" \
       --queryset "$query_set" --batch_size 8 -d 0 -v $victim_select

echo "-------------------------------------3----------------------------------------"
echo "----------------------------Training knockoff model---------------------------"
echo "------------------------------------------------------------------------------"

python ms/attacks/extraction/knockoff_nets_train_adv_pl.py "$output_model_knockoff_dir" \
       "$adv_arch"  "$dataset" --budgets "$budgets" --queryset "$query_set" -d 0 \
       --epochs "$epochs_knockoff" --lr 0.01 --lr_step 60 --scheduler_choice step --pretrained imagenet \
       --transferset_dir "$victim_base_dir"

echo "-------------------------------------4----------------------------------------"
echo "--------------------------Testing extraction success--------------------------"
echo "------------------------------------------------------------------------------"

python ms/utils/test.py "$victim_base_dir"  \
      -a "$output_model_knockoff_dir" \
      -v "$victim_select" -n -2

echo "-------------------------------------5----------------------------------------"
echo "-----------------Adversarial data construction----'$attack'-------------------"
echo "------------------------------------------------------------------------------"

python ms/attacks/adversarial/evasion.py "$attack" "$victim_base_dir" -v $victim_select \
      --epsilons "$eps"

echo "-------------------------------------6----------------------------------------"
echo "----------------------Adversarial Training----'$attack'-----------------------"
echo "------------------------------------------------------------------------------"
# Adversarial Training
# ratio
# 0: none adv examples
# 0.xx : xx% adv examples (100-xx)% original samples
# 1 : 100% adv samples
# 2 : 100% adv examples + 100% original samples

python ms/defences/train_robust.py "$victim_base_dir" \
       --out_dir "$victim_adv_robust_dir" \
       "$victim_robust_arch"  "$dataset" --ratio "$ratio" --attack "$attack" --epsilons "$eps_adv_robust" -d 0  \
       --log-interval 100 --epochs "$epochs" --lr 0.01 --scheduler_choice step --batch_size 256 --pretrained imagenet

echo "-------------------------------------7----------------------------------------"
echo "------------------Transfer set construction------robust-----------------------"
echo "------------------------------------------------------------------------------"

python ms/attacks/extraction/knockoff_nets_transfer.py $victim_adv_robust_dir \
       --out_dir "$output_model_knockoff_robust_dir" --budget "$budget" \
       --queryset "$query_set" --batch_size 32 -d 0 -v $victim_select

echo "-------------------------------------8----------------------------------------"
echo "-------------------Training knockoff model-----------robust-------------------"
echo "------------------------------------------------------------------------------"
##### Training knockoff model with a single budget --pretrained imagenet

python ms/attacks/extraction/knockoff_nets_train_adv_pl.py "$output_model_knockoff_robust_dir" \
       "$adv_arch"  "$dataset" --budgets "$budget" --queryset "$query_set" -d 0 --batch_size 256 \
       --epochs "$epochs_knockoff" --lr 0.01 --lr_step 60 --scheduler_choice step --suffix robust --epsilon 0.03

echo "-------------------------------------9----------------------------------------"
echo "-----------------Testing extraction success---------robust--------------------"
echo "------------------------------------------------------------------------------"

python ms/utils/test.py "$victim_adv_robust_dir"  \
       -a  "$output_model_knockoff_robust_dir" \
       -v "$victim_select" -n -2 -f 0.03
