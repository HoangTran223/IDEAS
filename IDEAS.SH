



ssh -p 8000 kc@label.bkict.org
Kc2024@!

unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES


--------------------------------------
20NG: --beta_temp 0.2 --weight_ECR 250 --dropout 0
Yahoo: --beta_temp 0.2 --weight_ECR 60 --dropout 0
AGNews: --beta_temp 0.1 --weight_ECR 5 --dropout 0
IMDB: --beta_temp 0.2 --weight_ECR 100 --dropout 0


TD_15: 0.98933
NMI:  0.40602729651533376
Purity:  0.8552
TC_15: 0.44731
python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 50 --alpha_ECR 18 --weight_loss_TP 2 --DT_alpha 2 --alpha_TP 3  \
--beta_temp 0.1 --dataset AGNews --model IDEAS --num_topics 50 --dropout 0.2 --seed 1 \
--lr 0.002 --epochs 363 --weight_loss_DT_ETP 0.5 --weight_loss_cl_large 1 --threshold_cluster 150 \
--num_large_clusters 9  --threshold_epochs 70 --weight_loss_cl_words 1 &&


TD_15: 0.87133
NMI:  0.3680341356157369
Purity:  0.8556
TC_15: 0.44633
python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 40 --alpha_ECR 18 --weight_loss_TP 2 --DT_alpha 2 --alpha_TP 3  \
--beta_temp 0.1 --dataset AGNews --model IDEAS --num_topics 100 --dropout 0.2 --seed 1 \
--lr 0.002 --epochs 400 --weight_loss_DT_ETP 1 --weight_loss_cl_large 1 --threshold_cluster 70 \
--num_large_clusters 17 --threshold_epochs 70 --weight_loss_cl_words 1 &&



TD_15: 0.91000
NMI:  0.3699101506268482
Purity:  0.8508
TC_15: 0.43911
python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 40 --alpha_ECR 18 --weight_loss_TP 2 --DT_alpha 2 --alpha_TP 3  \
--beta_temp 0.1 --dataset AGNews --model IDEAS --num_topics 100 --dropout 0.2 --seed 1 \
--lr 0.002 --epochs 400 --weight_loss_DT_ETP 2 --weight_loss_cl_large 2 --threshold_cluster 70 \
--num_large_clusters 17 --threshold_epochs 70 --weight_loss_cl_words 1 &&


python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 40 --alpha_ECR 18 --weight_loss_TP 2 --DT_alpha 2 --alpha_TP 3  \
--beta_temp 0.1 --dataset AGNews --model IDEAS --num_topics 100 --dropout 0.2 --seed 1 \
--lr 0.002 --epochs 20 --weight_loss_DT_ETP 4 --weight_loss_cl_large 2 --threshold_cluster 5 \
--num_large_clusters 17 --threshold_epochs 5 --weight_loss_cl_words 1






-----------------------------------------------------------------------
5:


TD_15: 0.86800
NMI:  0.5849958187317342
Purity:  0.6242697822623473
TC_15: 0.44454
python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 220 --alpha_ECR 20 --weight_loss_TP 2 --DT_alpha 2 --alpha_TP 2  \
--beta_temp 0.1 --dataset 20NG --model IDEAS --num_topics 50 --dropout 0.0 --seed 2 \
--lr 0.002 --epochs 400 --weight_loss_DT_ETP 1 --weight_loss_cl_large 1 --threshold_cluster 150 \
--num_large_clusters 9  --threshold_epochs 150 --weight_loss_cl_words 1 &&


TD_15: 0.89733
NMI:  0.07458521256583134
Purity:  0.7196
TC_15: 0.38762
python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 120 --alpha_ECR 20 --weight_loss_TP 1 --DT_alpha 3 --alpha_TP 3  \
--beta_temp 0.1 --dataset IMDB --model IDEAS --num_topics 50 --dropout 0.0 --seed 2 \
--lr 0.002 --epochs 350 --weight_loss_DT_ETP 2 --weight_loss_cl_large 2 --threshold_cluster 70 \
--num_large_clusters 9  --threshold_epochs 70 --weight_loss_cl_words 1


TD_15: 0.84533
NMI:  0.08130727924156275
Purity:  0.7348
TC_15: 0.40216
python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 123 --alpha_ECR 20 --weight_loss_TP 1 --DT_alpha 3 --alpha_TP 3  \
--beta_temp 0.1 --dataset IMDB --model IDEAS --num_topics 50 --dropout 0.0 --seed 2 \
--lr 0.002 --epochs 350 --weight_loss_DT_ETP 2 --weight_loss_cl_large 2 --threshold_cluster 70 \
--num_large_clusters 9  --threshold_epochs 70 --weight_loss_cl_words 1


TD_15: 0.71267
NMI:  0.5553829264095068
Purity:  0.6229421136484333
TC_15: 0.43115
python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 220 --alpha_ECR 20 --weight_loss_TP 2 --DT_alpha 2 --alpha_TP 2  \
--beta_temp 0.1 --dataset 20NG --model IDEAS --num_topics 100 --dropout 0.0 --seed 2 \
--lr 0.002 --epochs 405 --weight_loss_DT_ETP 1 --weight_loss_cl_large 1 --threshold_cluster 150 \
--num_large_clusters 9  --threshold_epochs 150 --weight_loss_cl_words 1



TD_15: 0.51267
NMI:  0.08748321916110392
Purity:  0.77396
TC_15: 0.38715
python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 100 --alpha_ECR 20 --weight_loss_TP 1 --DT_alpha 3 --alpha_TP 3  \
--beta_temp 0.1 --dataset IMDB --model IDEAS --num_topics 100 --dropout 0.0 --seed 2 \
--lr 0.002 --epochs 350 --weight_loss_DT_ETP 2 --weight_loss_cl_large 2 --threshold_cluster 70 \
--num_large_clusters 13 --threshold_epochs 70 --weight_loss_cl_words 1 &&



TD_15: 0.74133
NMI:  0.5678462751040536
Purity:  0.651619755708975
TC_15: 0.42439
python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 220 --alpha_ECR 20 --weight_loss_TP 2 --DT_alpha 2 --alpha_TP 2  \
--beta_temp 0.1 --dataset 20NG --model IDEAS --num_topics 100 --dropout 0.1 --seed 2 \
--lr 0.002 --epochs 405 --weight_loss_DT_ETP 1 --weight_loss_cl_large 1 --threshold_cluster 150 \
--num_large_clusters 9  --threshold_epochs 150 --weight_loss_cl_words 1 &&




TD_15: 0.73200
NMI:  0.32117755616746546
Purity:  0.5812
TC_15: 0.41967
python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 50 --alpha_ECR 20 --weight_loss_TP 2 --DT_alpha 2 --alpha_TP 2  \
--beta_temp 0.1 --dataset YahooAnswers --model IDEAS --num_topics 50 --dropout 0.2 --seed 3 \
--lr 0.002 --epochs 400 --weight_loss_DT_ETP 1 --weight_loss_cl_large 1 --threshold_cluster 100 \
--num_large_clusters 8 --threshold_epochs 100 --weight_loss_cl_words 1 &&



------------------------------------------------------------------------------------
3:



TD_15: 0.77933
NMI:  0.8637519263928577
Purity:  0.759687811961158
TC_15: 0.46863
python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 10 --alpha_ECR 20 --weight_loss_TP 2 --DT_alpha 2 --alpha_TP 2  \
--beta_temp 0.1 --dataset GoogleNews --model IDEAS --num_topics 100 --dropout 0.2 --seed 3 \
--lr 0.002 --epochs 400 --weight_loss_DT_ETP 1 --weight_loss_cl_large 1 --threshold_cluster 100 \
--num_large_clusters 15 --threshold_epochs 100 --weight_loss_cl_words 1



python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 10 --alpha_ECR 20 --weight_loss_TP 2 --DT_alpha 2 --alpha_TP 2  \
--beta_temp 0.1 --dataset YahooAnswers --model IDEAS --num_topics 100 --dropout 0.2 --seed 3 \
--lr 0.002 --epochs 400 --weight_loss_DT_ETP 1 --weight_loss_cl_large 1 --threshold_cluster 100 \
--num_large_clusters 15 --threshold_epochs 100 --weight_loss_cl_words 1 &&

python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 10 --alpha_ECR 20 --weight_loss_TP 2 --DT_alpha 2 --alpha_TP 2  \
--beta_temp 0.15 --dataset YahooAnswers --model IDEAS --num_topics 100 --dropout 0.2 --seed 34 \
--lr 0.002 --epochs 400 --weight_loss_DT_ETP 1 --weight_loss_cl_large 1 --threshold_cluster 100 \
--num_large_clusters 15 --threshold_epochs 100 --weight_loss_cl_words 1 &&

python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 10 --alpha_ECR 20 --weight_loss_TP 2 --DT_alpha 2 --alpha_TP 2  \
--beta_temp 0.1 --dataset YahooAnswers --model IDEAS --num_topics 100 --dropout 0.2 --seed 3 \
--lr 0.0014 --epochs 400 --weight_loss_DT_ETP 1 --weight_loss_cl_large 1 --threshold_cluster 100 \
--num_large_clusters 15 --threshold_epochs 100 --weight_loss_cl_words 1 &&

python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 10 --alpha_ECR 20 --weight_loss_TP 2 --DT_alpha 2 --alpha_TP 4  \
--beta_temp 0.1 --dataset YahooAnswers --model IDEAS --num_topics 100 --dropout 0.15 --seed 3 \
--lr 0.002 --epochs 400 --weight_loss_DT_ETP 1 --weight_loss_cl_large 1 --threshold_cluster 100 \
--num_large_clusters 15 --threshold_epochs 100 --weight_loss_cl_words 1 &&

python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 10 --alpha_ECR 20 --weight_loss_TP 2 --DT_alpha 2 --alpha_TP 2  \
--beta_temp 0.1 --dataset YahooAnswers --model IDEAS --num_topics 100 --dropout 0.2 --seed 3 \
--lr 0.002 --epochs 400 --weight_loss_DT_ETP 1 --weight_loss_cl_large 1 --threshold_cluster 200 \
--num_large_clusters 19 --threshold_epochs 200 --weight_loss_cl_words 1 &&

python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 10 --alpha_ECR 20 --weight_loss_TP 2 --DT_alpha 2 --alpha_TP 2  \
--beta_temp 0.1 --dataset YahooAnswers --model IDEAS --num_topics 100 --dropout 0.2 --seed 24 \
--lr 0.002 --epochs 400 --weight_loss_DT_ETP 1 --weight_loss_cl_large 1 --threshold_cluster 100 \
--num_large_clusters 15 --threshold_epochs 100 --weight_loss_cl_words 1 &&

python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 10 --alpha_ECR 20 --weight_loss_TP 2 --DT_alpha 2 --alpha_TP 2  \
--beta_temp 0.1 --dataset YahooAnswers --model IDEAS --num_topics 100 --dropout 0.2 --seed 6102004 \
--lr 0.002 --epochs 400 --weight_loss_DT_ETP 2 --weight_loss_cl_large 1 --threshold_cluster 150 \
--num_large_clusters 15 --threshold_epochs 150 --weight_loss_cl_words 1




--------------------------------------------------------------------------------
4:

TD_15: 1.00000
NMI:  0.4662907264459663
Purity:  0.8062469497315764
TC_15: 0.45830
python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 40 --alpha_ECR 20 --weight_loss_TP 2 --DT_alpha 2 --alpha_TP 2  \
--beta_temp 0.1 --dataset SearchSnippets --model IDEAS --num_topics 50 --dropout 0.2 --seed 4 \
--lr 0.002 --epochs 400 --weight_loss_DT_ETP 1 --weight_loss_cl_large 1 --threshold_cluster 200 \
--num_large_clusters 8  --threshold_epochs 200 --weight_loss_cl_words 1 &&



TD_15: 0.96000
NMI:  0.7118797577553246
Purity:  0.46084036663944095
TC_15: 0.45717
python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 40 --alpha_ECR 20 --weight_loss_TP 2 --DT_alpha 2 --alpha_TP 2  \
--beta_temp 0.1 --dataset GoogleNews --model IDEAS --num_topics 50 --dropout 0.2 --seed 4 \
--lr 0.002 --epochs 400 --weight_loss_DT_ETP 1 --weight_loss_cl_large 1 --threshold_cluster 200 \
--num_large_clusters 8  --threshold_epochs 200 --weight_loss_cl_words 1

python main.py 
--device cuda \
--lr_scheduler StepLR \
--batch_size 128 \
--lr_step_size 125 \
--use_pretrainWE \
--weight_ECR 40 \
--alpha_ECR 20 \
--weight_loss_TP 2 \
--DT_alpha 2 \
--alpha_TP 2  \
--beta_temp 0.1 \
--dataset GoogleNews \
--model IDEAS \
--num_topics 50 \
--dropout 0.2 \
--seed 4 \
--lr 0.002 \
--epochs 400 \
--weight_loss_DT_ETP 1 \
--weight_loss_cl_large 1 \
--threshold_cluster 200 \
--num_large_clusters 8  \
--threshold_epochs 200 \
--weight_loss_cl_words 1





TD_15: 0.96800
NMI:  0.482328630931395
Purity:  0.8430941922889215
TC_15: 0.44604
python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 40 --alpha_ECR 20 --weight_loss_TP 2 --DT_alpha 2 --alpha_TP 2  \
--beta_temp 0.1 --dataset SearchSnippets --model IDEAS --num_topics 100 --dropout 0.2 --seed 4 \
--lr 0.0023 --epochs 400 --weight_loss_DT_ETP 3 --weight_loss_cl_large 1 --threshold_cluster 200 \
--num_large_clusters 15  --threshold_epochs 200 --weight_loss_cl_words 1 &&


TD_15: 0.72667
NMI:  0.07017635496026489
Purity:  0.73556
TC_15: 0.38860
python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 100 --alpha_ECR 20 --weight_loss_TP 1 --DT_alpha 3 --alpha_TP 3  \
--beta_temp 0.15 --dataset IMDB --model IDEAS --num_topics 100 --dropout 0.0 --seed 4 \
--lr 0.002 --epochs 350 --weight_loss_DT_ETP 2 --weight_loss_cl_large 2 --threshold_cluster 70 \
--num_large_clusters 13 --threshold_epochs 70 --weight_loss_cl_words 1 &&

python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 100 --alpha_ECR 20 --weight_loss_TP 1 --DT_alpha 3 --alpha_TP 3  \
--beta_temp 0.1 --dataset IMDB --model IDEAS --num_topics 100 --dropout 0.0 --seed 4 \
--lr 0.0017 --epochs 350 --weight_loss_DT_ETP 2 --weight_loss_cl_large 2 --threshold_cluster 70 \
--num_large_clusters 13 --threshold_epochs 70 --weight_loss_cl_words 1 &&

python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 100 --alpha_ECR 20 --weight_loss_TP 1 --DT_alpha 3 --alpha_TP 3  \
--beta_temp 0.1 --dataset IMDB --model IDEAS --num_topics 100 --dropout 0.0 --seed 4 \
--lr 0.002 --epochs 320 --weight_loss_DT_ETP 2 --weight_loss_cl_large 2 --threshold_cluster 100 \
--num_large_clusters 13 --threshold_epochs 100 --weight_loss_cl_words 1 &&

python main.py --device cuda --lr_scheduler StepLR --batch_size 128 --lr_step_size 125 \
--use_pretrainWE --weight_ECR 100 --alpha_ECR 20 --weight_loss_TP 1 --DT_alpha 3 --alpha_TP 3  \
--beta_temp 0.1 --dataset IMDB --model IDEAS --num_topics 100 --dropout 0.0 --seed 4 \
--lr 0.002 --epochs 400 --weight_loss_DT_ETP 1 --weight_loss_cl_large 2 --threshold_cluster 70 \
--num_large_clusters 13 --threshold_epochs 70 --weight_loss_cl_words 3 &&



--------------------------------------------------------------------------------

2: 



