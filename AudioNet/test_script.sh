python evaluate.py \
--vertex_file_path /viscam/u/rhgao/datasets/ObjectFiles/YCB/ycb/030_fork/google_64k/vertices.npy \
--modes_file_path /viscam/u/rhgao/datasets/ObjectFiles/YCB/ycb/030_fork/google_64k/modes.npy \
--model_path ./checkpoints/exp2/model.pt \
--gpu_ids 0,1 \
--batchSize 10000 \
--network_depth 8 \
--results_dir ./results/exp2/
