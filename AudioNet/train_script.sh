python main.py \
--vertex_file_path /viscam/u/rhgao/datasets/ObjectFiles/modelhaven/13/vertices.npy \
--modes_file_path /viscam/u/rhgao/datasets/ObjectFiles/modelhaven/13/modes.npy \
--gpu_ids 0,1,2,3 \
--display_freq 20 \
--batchSize 50000 \
--iterations 100000 \
--tensorboard True
