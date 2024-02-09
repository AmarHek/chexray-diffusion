model:
base_learning_rate: float
target: path to lightning module
params:
    key: value
ta:
target: main.DataModuleFromConfig
params:
   batch_size: int
   wrap: bool
   train:
       target: path to train dataset
       params:
           key: value
   validation:
       target: path to validation dataset
       params:
           key: value
   test:
       target: path to test dataset
       params:
           key: value
ghtning: (optional, has sane defaults and can be specified on cmdline)
trainer:
    additional arguments to trainer
logger:
    logger to instantiate
modelcheckpoint:
    modelcheckpoint to instantiate
callbacks:
    callback1:
        target: importpath
        params:
            key: value