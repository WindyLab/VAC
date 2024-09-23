
config_train = {
    'batch_size': 32,
    'epochs': 200,
    'save_epoch': 20,
    'learning_rate': 3*1e-06,
    'weight_decay': 1e-5,
    'momentum': 0.999,
    'amp': False,
    'lr_scheduler': 'step1',
    'val_percent': 0.1,
    'load_pretrained': False,
    'data_id':0,
    'model_dir': 'checkpoints/va/Model1.pth',
    'yolo_dir': 'checkpoints/yolo/best_real_close.pt',
    'checkpoint_dir':'checkpoints',
    'with_motion': False,
    'horizontal_fov': 2,
    'va_channel': [32,64,128,256,512],
    'bilinear': False
}
