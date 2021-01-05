import keras.callbacks

def tune_schedule(lr=1e-3):
    def step_decay(epoch):
        if epoch < 15:
            return lr
        if epoch < 25:
            return lr*0.1
        if epoch < 30:
            return lr*0.01
        else:
            return lr*0.001
    return step_decay

def TuneScheduler(lr=1e-3):
    return keras.callbacks.LearningRateScheduler(tune_schedule(lr))
