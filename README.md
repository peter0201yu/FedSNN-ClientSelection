# Client Selection for Federated Learning with Spiking Neural Networks

## Acknowledgements

Code adapted from [Federated Learning with Spiking Neural Networks](https://github.com/Intelligent-Computing-Lab-Yale/FedSNN).


## Environment

See `environment.yml`.


## Client selection strategies

In `models/client_selection.py`, I implemented a few client selection strategies based on previous research papers. 

- Random: select clients randomly.

- Biggest loss: select clients with biggest local forward loss when forwarding data on newly received global model. Based on this paper: [Client Selection in Federated Learning: Convergence Analysis and Power-of-Choice Selection Strategies](https://arxiv.org/abs/2010.01243).

- Biggest training loss: select clients with biggest local training loss.

- Gradient diversity: select a group of clients whose weighted sum of gradients (delta weight) best approximates the sum of gradients of all clients. Based on this paper: [Diverse Client Selection for Federated Learning: Submodularity and Convergence Analysis](https://fl-icml.github.io/2021/papers/FL-ICML21_paper_67.pdf).

- Update norm: select clients with biggest update/gradient norms. Based on this paper: [Optimal Client Sampling for Federated Learning](https://arxiv.org/abs/2010.13723).

- Spike activity diversity: select clients with the most diverse spike activities (only applicable to SNNs).


## Candidate selection strategies

In `models/candidate_selection.py`, I implemented a few candidate selection strategies. The reason for selecting candidates is that most client selection algorithms require the local training information of clients. Our simulated federated learning system, however, cannot handle the training of many clients in each round. Therefore, we first select ~20 candidates to train and then select among them for models to be uploaded and aggregated.

- Random: select candidates randomly.

- Loop: select candidates in a loop, no candidate overlap in consecutive rounds.

- Data amount: candidates with more data are more likely to be selected.

- Reduce collision: candidates that were previously chosen become less likely to be chosen.


## Heterogeneous training and weighted FedAvg strategies

In `heterogeneous.py`, we explore the potential of performing federated learning with SNNs on heterogeneous devices. More specifically, we can adjust the training timesteps based on the device's computing power, and when aggregating the model, we can perform FedAvg with different weight for the models.

The numbers of timesteps of the models are generated using the `timestep_mean` and `timestep_std` arguments. I also implemented a few weighted FedAvg strategies (inside the script):

- timestep_prop: the weighting coefficient of the model is proportional to the number of timesteps used for training the model

- timestep_inv: the weighting is inversely proportional to the number of timesteps

- train_loss_prop: the weighting is proportional to the training loss

- train_loss_inv: the weighting is inversely proportional to the training loss


## Experiments

`single_model.py` trains and evaluates a single model (without federated learning) while using a portion of the total data (to imitate the amount of local training data in federated learning). When trying out a new set of hyperparameters, run this script to separate the strategy's effect on local training and the strategy's effect on federated learning. `test_single.sh` contains an example that runs the script with arguments.

`client_experiment.py` contains the components needed to compare client/candidate selection strategies. `test_cifar10_clients.sh` contains an example that runs the script with arguments.

`heterogeneous.py` contains the components needed to compare weighted FedAvg strategies in the heterogeneous training scenario. `test_cifar_hetero.sh` contains an example that runs the script with arguments.


## Results - wandb projects

wandb project for [client/candidate selection experiments](https://wandb.ai/peteryu/FedSNN-candidate?workspace=user-peteryu)

wandb project for [heterogeneous training experiments](https://wandb.ai/peteryu/FedSNN-heterogeneous?workspace=user-peteryu)
