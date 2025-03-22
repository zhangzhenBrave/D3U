
seq_len=96
pred_len=192
model_name=SVQ

root_path_name='../../../dataset/Solar/'
data_path_name=solar_AL.txt
model_id_name=solar
data_name='Solar'

random_seed=2021
python -u ../../../runner9_NS_transformer.py \
        --seed $random_seed \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name_$seq_len'_'$pred_len \
        --model $model_name \
        --data_name $model_id_name\
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --label_len 48 \
        --pred_len $pred_len \
        --enc_in 137\
        --dec_in 137\
        --c_out  137\
        --e_layers_c 2 \
        --n_heads_c 8 \
        --d_model_c 256 \
        --d_ff 512\
        --dropout 0.2\
        --fc_dropout 0.2\
        --head_dropout 0\
        --depth 1\
        --d_model_d 128\
        --num_workers 4\
        --itr 1\
        --train_epochs 100\
        --timesteps 100\
        --batch_size 16\
        --test_batch_size 16\
        --des 'Exp'\
        --lradj 'type1'\
        --denoise_model 'PatchDN'\
        --kernel_size 15\
        --fourier_factor 1.0\
        --svq 1 \
        --wFFN 0 \
        --num_codebook 1\
        --codebook_size 1000 \
        --type_sample 'DPM_solver'\
        --DPMsolver_step 20\
        --gpu 0 \
        --parameterization "x_start"\
        --bias \