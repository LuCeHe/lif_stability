def _get_cifar10_dataset(cfg, train=True, valid=True, test=True):

    logger.info("Loading CIFAR10".format(cfg.nb_classes))
    
    all_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        torchvision.transforms.Lambda(lambda tensor: tensor.reshape((1, 3, 32, 32))),
        torchvision.transforms.Lambda(lambda tensor: tensor.expand((cfg.nb_time_steps, -1, -1, -1))),
        ])
    
    test_transforms = all_transforms
    train_transforms = all_transforms
        
    if train:
        ds_train = torchvision.datasets.CIFAR10(cfg.path, 
                                                    train=True, 
                                                    download=True, 
                                                    transform=train_transforms)
        logger.info("Generated {} training data".format(len(ds_train)))
        
    else:
        ds_train = False
    
    if valid:
        split_lengths = [int(len(ds_train) * (1 - cfg.valid_split)), int(len(ds_train) * cfg.valid_split)]
        ds_train, ds_valid = torch.utils.data.dataset.random_split(ds_train, split_lengths)
        logger.info("Generated {} validation data".format(len(ds_valid)))
        
    else:
        ds_valid = False

    if test: 
        ds_test = torchvision.datasets.CIFAR10(cfg.path, 
                                                    train=False, 
                                                    download=True, 
                                                    transform=test_transforms)
        logger.info("Generated {} testing data".format(len(ds_test)))

    else:
        ds_test = False
            
    return {"train": ds_train, "valid": ds_valid, "test": ds_test}


def _get_DVSgestures_dataset(cfg, train=True, valid=True, test=True):
    
    target_size = cfg.nb_inputs[-1]
    
    # Transforms
    
    # Drop random events
    tf_dropevent = tonic.transforms.DropEvent(p = cfg.dropevent_p)

    # Convert to milliseconds
    tf_convert_to_ms = tonic.transforms.Downsample(time_factor = 1e-3, 
                                                        spatial_factor= target_size/128)

    # Assemble frames according to timestep
    tf_frame = tonic.transforms.ToFrame(sensor_size=(target_size, target_size, 2),
                                        time_window=cfg.time_step*1000)
    
    # CUSTOM TRANSFORMS
    
    class ToTensorTransform:
        """ Custom ToTensor transform that supports 4D arrays"""
        def __init__(self, bool_spiketrain=False):
            self.bool_spiketrain = bool_spiketrain

        def __call__(self, x):
            if self.bool_spiketrain:
                return torch.as_tensor(x).bool().float()
            else:
                return torch.as_tensor(x).float()
                
    tf_tensor = ToTensorTransform(cfg.bool_spiketrain)
 
    class TimeCropTransform:
        """ Custom transform that randomly crops the time dimension"""
        def __init__(self, timesteps):
            self.timesteps = int(timesteps)

        def __call__(self, x):
            start = np.random.randint(0, high=x.shape[0]-self.timesteps)
            return x[start:start+self.timesteps, :, :, :]
    
    tf_timecrop = TimeCropTransform(cfg.nb_time_steps)

    all_transforms = tonic.transforms.Compose([tf_dropevent,
                                               tf_convert_to_ms,
                                               tf_frame, 
                                               tf_tensor,
                                               tf_timecrop])
        
    if train:
        ds_train = tonic.datasets.DVSGesture(cfg.path,
                                             train=True, 
                                             transform=all_transforms)
        
        logger.info("Generated {} training data".format(len(ds_train)))
        
    else:
        ds_train = False
    
    ds_valid=False

    if test: 
        ds_test = tonic.datasets.DVSGesture(cfg.path, 
                                            train=False,
                                            transform=all_transforms)
        
        logger.info("Generated {} testing data".format(len(ds_test)))

    else:
        ds_test = False
            
    return {"train": ds_train, "valid": ds_valid, "test": ds_test}