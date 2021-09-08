python main.py \
--vertex_file_path /viscam/u/rhgao/datasets/ObjectFiles/GoogleScannedObjects/Kanex_MultiSync_Wireless_Keyboard/TouchData/verts_forces.npy \
--touch_file_path /viscam/u/rhgao/datasets/ObjectFiles/GoogleScannedObjects/Kanex_MultiSync_Wireless_Keyboard/touch.npy \
--gpu_ids 0 \
--display_freq 20 \
--batchSize 50000 \
--iterations 100000 \
--tensorboard True
