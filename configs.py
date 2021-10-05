class ConfigParameters:
    def __init__(self):
        self.mode = "train"
        self.gpu_id = "0"
        self.train_continue = "off"

        self.num_modes = 3
        
        self.lr = 0.01
        self.batch_size = 100
        self.num_epoch = 50

        self.data_dir = ""
        self.ckpt_dir = "./checkpoint/"
        self.log_dir = "./log"
        self.result_dir = "./result"