python main_fed.py --snn --dataset EMNIST --num_classes 47 --img_size 28 --model simple --bntt --direct --optimizer SGD --bs 32 --local_bs 32 --lr 0.1 --epochs 30 --local_ep 2 --eval_every 1 --alpha 0.2 --train_frac 1 --test_size 10000 --num_users 500 --client_selection hybrid --frac 0.02 --candidate_selection reduce_collision --gamma 4 --candidate_frac 0.04 --gpu 0 --timesteps 10 --straggler_prob 0.0 --grad_noise_stdev 0.0 --result_dir emnist_500c20c10_hb_rc_gamma4 --project FedSNN-candidate --wandb emnist_500c20c10_hb_rc_gamma4
################################
# python main_fed.py --snn --dataset EMNIST --num_classes 47 --img_size 28 --model simple --bntt --direct --optimizer SGD --bs 32 --local_bs 32 --lr 0.1 --epochs 30 --local_ep 2 --eval_every 1 --alpha 0.2 --train_frac 1 --test_size 10000 --num_users 500 --client_selection grad_diversity --frac 0.02 --candidate_selection reduce_collision --gamma 4 --candidate_frac 0.04 --gpu 0 --timesteps 10 --straggler_prob 0.0 --grad_noise_stdev 0.0 --result_dir emnist_500c20c10_gd_rc_gamma4 --project FedSNN-candidate --wandb emnist_500c20c10_gd_rc_gamma4
################################
# python main_fed.py --snn --dataset EMNIST --num_classes 47 --img_size 28 --model simple --bntt --direct --optimizer SGD --bs 32 --local_bs 32 --lr 0.1 --epochs 30 --local_ep 2 --eval_every 1 --alpha 0.2 --train_frac 1 --test_size 10000 --num_users 500 --client_selection grad_diversity --frac 0.02 --candidate_selection reduce_collision --gamma 8 --candidate_frac 0.04 --gpu 0 --timesteps 10 --straggler_prob 0.0 --grad_noise_stdev 0.0 --result_dir emnist_500c20c10_gd_rc_gamma8 --project FedSNN-candidate --wandb emnist_500c20c10_gd_rc_gamma8