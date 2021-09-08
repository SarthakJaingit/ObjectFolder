# VisionNet

### Data Preparation
  * `modelhaven`: 1-20.
  * `GoogleScannedObjects`: 21-72.
  * `ycb`: 73-100.
```
  $ cd blender
  $ blender -b -P gpu.py -E CYCLES --python generate_{modelhaven/GoogleScannedObject/ycb}_views.py \
     -- --mesh_file path_of_obj_file \
     --mtl_file path_of_mtl_file \
     --results_path path_to_save_results
```
This code can be given with the following command-line arguments:
  * `--mesh_file`: the path of the obj file.
  * `--mtl_file`: the path of the mtl file.
  * `--results_path`: the path to save results.
  * `--radius`: the radius of the sphere, default is 2.5.
  * `--random_views`: randomly put the camera on the sphere or not, default is false.
  * `--upper_views`: put the camera on the upper hemisphere or not, default is false.

For example:
```
  $ blender -b -P gpu.py -E CYCLES --python generate_modelhaven_views.py \
       -- --mesh_file ../object/1/GothicCabinet_01.obj \
       --mtl_file ../object/1/GothicCabinet_01.mtl \
       --results_path ../object/1/data/train \
       --random_views 
```

### Training
```
  $ python run_osf.py --config configs/{1-100}.txt
```

### Testing
```
  $ python run_osf.py --config configs/{1-100}.txt --render_only --render_test --render_start 0 --render_end 1000
```
