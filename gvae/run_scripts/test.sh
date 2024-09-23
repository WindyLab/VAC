GPU=0
DATA_PATH='data/swarm/'
mkdir -p $DATA_PATH
BASE_RESULTS_DIR="results/swarm/"
SEED=19
WORKING_DIR="${BASE_RESULTS_DIR}/seed_${SEED}_2/"
ENCODER_ARGS='--num_edge_types 2 --encoder_hidden 256 --skip_first --prior_num_layers 2 --encoder_mlp_hidden 256 --encoder_mlp_num_layers 2'
DECODER_ARGS="--decoder_hidden 256"
MODEL_ARGS="--model_type dnri --graph_type static ${ENCODER_ARGS} ${DECODER_ARGS} --seed ${SEED}"
TRAINING_ARGS='--kl_coef 0.0001 --lr 1e-3 --use_adam --num_epochs 100 --lr_decay_factor 0.5 --lr_decay_steps 50 --normalize_kl --normalize_nll --tune_on_nll --val_teacher_forcing --teacher_forcing_steps -1'
mkdir -p $WORKING_DIR
CUDA_VISIBLE_DEVICES=$GPU python -u train_swarm.py --gpu --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS |& tee "${WORKING_DIR}results.txt"