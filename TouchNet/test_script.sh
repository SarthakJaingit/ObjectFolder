python evaluate.py \
--vertex_file_path /viscam/u/rhgao/datasets/ObjectFiles/GoogleScannedObjects/5_HTP/TouchData/verts_forces.npy \
--touch_file_path /viscam/u/rhgao/datasets/ObjectFiles/GoogleScannedObjects/5_HTP/touch.npy \
--results_path /viscam/u/rhgao/datasets/ObjectFiles/GoogleScannedObjects/5_HTP/touch_prediction.npy \
--model_path /viscam/u/rhgao/datasets/ObjectFiles/GoogleScannedObjects/5_HTP/5_HTP_TouchNet/model.pt \
--gpu_ids 0 \
--batchSize 10000 \
--network_depth 8 \
