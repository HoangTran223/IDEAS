# NeuroMax: Enhancing Neural Topic Modeling via Maximizing Mutual Information and Group Topic Regularization

## Preparing libraries
1. Install the following libraries
    ```
    numpy 1.26.4
    torch_kmeans 0.2.0
    pytorch 2.2.0
    sentence_transformers 2.2.2
    scipy 1.10
    bertopic 0.16.0
    gensim 4.2.0
    ```
2. Install java
3. Download [this java jar](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/palmetto-0.1.0-jar-with-dependencies.jar) to ./evaluations/pametto.jar
4. Download and extract [this processed Wikipedia corpus](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/Wikipedia_bd.zip) to ./datasets/wikipedia/ as an external reference corpus.

## Usage
To run and evaluate our model, run the following command:

> python main.py --model ETM --dataset YahooAnswers --num_topics 50 --beta_temp 0.15 --num_groups 20 --epoch_threshold 140 --epochs 200 --device cuda --lr 0.002 --lr_scheduler StepLR --dropout 0.2 --batch_size 200 --lr_step_size 125 --use_pretrainWE --weight_ECR 40 --weight_GR 1.0 --alpha_ECR 20.0 --alpha_GR 5.0 --weight_InfoNCE 50.0 --weight_OT 1 --rho 0.005 --lmbda 0.9 --use_SAM 1 --SAM_name DREAM

## Acknowledgement
Some part of this implementation is based on [TopMost](https://github.com/BobXWu/TopMost). We also utilizes [Palmetto](https://github.com/dice-group/Palmetto) for the evaluation of topic coherence.
