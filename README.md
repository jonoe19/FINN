# FINN

## Path Planning
To generate the path to traverse and make the drone follow it, execute 
```
python3 simulation.py
```

## Creating the artificial dataset
To generate the dataset, change the `image_dir` and `output_dir` on lines 173 and 174, and run 
```
python3 datagen.py
```

## Converting real images to compatible images
To convert real images to a format that is readable by the classifier, change the `image_dir` and `output_dir` on lines 39 and 40, and run 
```
python3 dataconv.py
```

## Running the classifier
To run the classifier, run
```
python3 main.py
```

## Testing the Neural Network
To test the neural network, choose the directories you want to test from on lines 19-25, and run
```
python3 test.py
```

## Open environment
To run the environment with the apropriate visuals it is needed to change the corresponding paths to materials and textures. 
To open gazebo world, run
```
gazebo-11.10.2 --pause /path/to/my_world.world
```

## Launch simulation
To launch the simulation run,
```
roslaunch path/to/my_world.launch
```
Due to the unavailability of packages in Ubuntu 22.04, ROS could not be succesfully installed. However, this command will call an untested  `script goto_point.py` which should initialize the simulation in environment `my_world.world`.
