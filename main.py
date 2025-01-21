from utils import config, log, miscellaneous, seed
import os
import numpy as np
import basic_trainer
from NeuroMax.NeuroMax import NeuroMax
from FASTopic.FASTopic import FASTopic
from ECRTM.ECRTM import ECRTM
from ETM.ETM import ETM
import evaluations
import datasethandler
import scipy
import torch
import h5py
from tqdm import tqdm

RESULT_DIR = 'results'
DATA_DIR = 'datasets'

def get_model_params_vector(model):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    return torch.cat(params)

def set_model_params_vector(model, vector):
    pointer = 0
    for param in model.parameters():
        num_param = param.numel() 
        param.data.copy_(vector[pointer:pointer + num_param].view(param.size()))
        pointer += num_param

def random_directions(param_vector):
    direction1 = torch.randn_like(param_vector)
    direction2 = torch.randn_like(param_vector)
    return direction1, direction2


if __name__ == "__main__":
    parser = config.new_parser()
    config.add_dataset_argument(parser)
    config.add_model_argument(parser)
    config.add_training_argument(parser)
    args = parser.parse_args()

    current_time = miscellaneous.get_current_datetime()
    current_run_dir = os.path.join(RESULT_DIR, current_time)
    miscellaneous.create_folder_if_not_exist(current_run_dir)

    config.save_config(args, os.path.join(current_run_dir, 'config.txt'))
    seed.seedEverything(args.seed)
    print(args)

    if args.dataset in ['YahooAnswers', '20NG', 'AGNews', 'IMDB', 'SearchSnippets', 'StackOverflow', 'GoogleNews']:
        read_labels = True
    else:
        read_labels = False
    print(f"read labels = {read_labels}")
    cluster_distribution = np.load(os.path.join(DATA_DIR, str(args.dataset), "LLM", "cluster_distribution.npz"))['arr_0']
    cluster_mean = np.load(os.path.join(DATA_DIR, str(args.dataset), "LLM", "cluster_mean.npz"))['arr_0']
    cluster_label = [np.argmax(cluster_distribution[i]) for i in range(len(cluster_distribution))]

    dataset = datasethandler.BasicDatasetHandler(
        os.path.join(DATA_DIR, args.dataset), device=args.device, read_labels=read_labels,
        as_tensor=True, contextual_embed=True)

    pretrainWE = scipy.sparse.load_npz(os.path.join(
        DATA_DIR, args.dataset, "word_embeddings.npz")).toarray()
    
    if args.model == 'NeuroMax':
        model = NeuroMax(vocab_size=dataset.vocab_size,
                        data_name=args.dataset,
                        num_topics=args.num_topics,
                        num_groups=args.num_groups,
                        dropout=args.dropout,
                        cluster_distribution=cluster_distribution,
                        cluster_mean=cluster_mean,
                        cluster_label=cluster_label,
                        pretrained_WE=pretrainWE if args.use_pretrainWE else None,
                        weight_loss_GR=args.weight_GR,
                        weight_loss_ECR=args.weight_ECR,
                        alpha_ECR=args.alpha_ECR,
                        alpha_GR=args.alpha_GR,
                        weight_loss_OT=args.weight_OT,
                        weight_loss_InfoNCE=args.weight_InfoNCE,
                        beta_temp=args.beta_temp)
    elif args.model == 'FASTopic':
        model = FASTopic(vocab_size=dataset.vocab_size,
                        embed_size=dataset.contextual_embed_size,
                        num_topics=args.num_topics,
                        cluster_distribution=cluster_distribution,
                        cluster_mean=cluster_mean,
                        cluster_label=cluster_label,
                        weight_loss_OT=args.weight_OT)
    elif args.model == 'ECRTM':
        model = ECRTM(vocab_size=dataset.vocab_size,
                        num_topics=args.num_topics,
                        dropout=args.dropout,
                        cluster_distribution=cluster_distribution,
                        cluster_mean=cluster_mean,
                        cluster_label=cluster_label,
                        pretrained_WE=pretrainWE if args.use_pretrainWE else None,
                        weight_loss_ECR=args.weight_ECR,
                        alpha_ECR=args.alpha_ECR,
                        weight_OT=args.weight_OT,
                        beta_temp=args.beta_temp)
    elif args.model == 'ETM':
        model = ETM(vocab_size=dataset.vocab_size,
                        num_topics=args.num_topics,
                        dropout=args.dropout,
                        cluster_distribution=cluster_distribution,
                        cluster_mean=cluster_mean,
                        cluster_label=cluster_label,
                        pretrained_WE=pretrainWE if args.use_pretrainWE else None,
                        weight_OT=args.weight_OT
                        )

    
    model.weight_loss_GR = args.weight_GR
    model.weight_loss_ECR = args.weight_ECR
    model = model.to(args.device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")

    # create a trainer
    trainer = basic_trainer.BasicTrainer(model, model_name=args.model,
                                            epoch_threshold = args.epoch_threshold,
                                            use_SAM = args.use_SAM,
                                            SAM_name=args.SAM_name, epochs=args.epochs,
                                            learning_rate=args.lr,
                                            rho=args.rho,
                                            batch_size=args.batch_size,
                                            lr_scheduler=args.lr_scheduler,
                                            lr_step_size=args.lr_step_size,
                                            device=args.device,
                                            sigma=args.sigma,
                                            lmbda=args.lmbda
                                            )


    # train the model
    trainer.train(dataset)

    if args.render == 1:
        torch.save(model, args.model + args.dataset + str(args.use_SAM) + str(args.SAM_name) +'.pth')
        model.to('cuda')
        losses = np.zeros((args.step, args.step))
        alpha_vals = np.linspace(-args.alpha_range, args.alpha_range, args.step)
        beta_vals = np.linspace(-args.alpha_range, args.alpha_range, args.step)

        param_vector = get_model_params_vector(model)
        direction1, direction2 = random_directions(param_vector)

        model.theta_train = True
        for i, alpha in tqdm(enumerate(alpha_vals)):
            for j, beta in enumerate(beta_vals):
                new_params = param_vector + alpha * direction1 + beta * direction2
                set_model_params_vector(model, new_params)
                total_loss = 0
                data_size = dataset.train_data.shape[0]
                all_idx = torch.split(torch.arange(data_size), args.batch_size)
                with torch.no_grad():
                    model.eval()
                    loss_ = 0
                    for batch_id, batch in enumerate(dataset.train_dataloader):
                        *inputs, indices = batch
                        batch_data = inputs
                        rst_dict = model(indices, batch_data, epoch_id=0)
                        loss_ += rst_dict['loss']
                losses[i, j] = loss_ / len(dataset.train_dataloader)
        model.theta_train = False
        np.savez('loss_landscape/' + args.model + args.dataset + str(args.use_SAM) + str(args.SAM_name) + 'loss_landscape.npz', alpha_vals=alpha_vals, beta_vals=beta_vals, losses=losses)
        np.savetxt('loss_landscape/' + args.dataset + str(args.use_SAM) + str(args.SAM_name) + 'alpha_vals.txt', alpha_vals, fmt='%.6f')
        np.savetxt('loss_landscape/' + args.dataset + str(args.use_SAM) + str(args.SAM_name) + 'beta_vals.txt', beta_vals, fmt='%.6f')
        np.savetxt('loss_landscape/' + args.dataset + str(args.use_SAM) + str(args.SAM_name) + 'losses.txt', losses, fmt='%.6f')
        '''with h5py.File(args.model + args.dataset + str(args.use_SAM) + str(args.SAM_name) +'.h5', 'w') as f:
            for name, param in model.state_dict().items():
                f.create_dataset(name, data=param.cpu().numpy())'''

    # save beta, theta and top words
    beta = trainer.save_beta(current_run_dir)
    train_theta, test_theta = trainer.save_theta(dataset, current_run_dir)
    top_words_10 = trainer.save_top_words(
        dataset.vocab, 10, current_run_dir)
    top_words_15 = trainer.save_top_words(
        dataset.vocab, 15, current_run_dir)
    top_words_20 = trainer.save_top_words(
        dataset.vocab, 20, current_run_dir)
    top_words_25 = trainer.save_top_words(
        dataset.vocab, 25, current_run_dir)

    # argmax of train and test theta
    train_theta_argmax = train_theta.argmax(axis=1)
    test_theta_argmax = test_theta.argmax(axis=1)        

    TD_15 = evaluations.compute_topic_diversity(
        top_words_15, _type="TD")
    print(f"TD_15: {TD_15:.5f}")


    # evaluating clustering
    if read_labels:
        clustering_results = evaluations.evaluate_clustering(
            test_theta, dataset.test_labels)
        print(f"NMI: ", clustering_results['NMI'])
        print(f'Purity: ', clustering_results['Purity'])


    TC_15_list, TC_15 = evaluations.topic_coherence.TC_on_wikipedia(
        os.path.join(current_run_dir, 'top_words_15.txt'))
    print(f"TC_15: {TC_15:.5f}")

    filename = f"results_{args.dataset}_topics{args.num_topics}_epochs{args.epochs}_w_ECR{args.weight_ECR}_w_GR{args.weight_GR}_w_OT{args.weight_OT}_w_InfoNCE{args.weight_InfoNCE}.txt"
    filename = filename.replace(' ', '_')
    filepath = os.path.join(current_run_dir, filename)
    with open(filepath, 'w') as f:
        if read_labels:
            f.write(f"NMI: {clustering_results['NMI']}\n")
            f.write(f"Purity: {clustering_results['Purity']}\n")
        else:
            f.write("NMI: N/A\n")
            f.write("Purity: N/A\n")
        f.write(f"TD_15: {TD_15:.5f}\n")
        f.write(f"TC_15: {TC_15:.5f}\n")

    print(f"Done in {filepath}")

