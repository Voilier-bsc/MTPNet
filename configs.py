class ConfigParameters:
    def __init__(self):
        self.mode = "train"
        self.gpu_id = "0"
        self.train_continue = "off"

        self.num_modes = 3
        
        self.lr = 0.01
        self.batch_size = 4
        self.num_epoch = 520

        # self.data_dir = '/home/mmc-server3/Server/dataset/NuScenes'
        self.data_dir = './dataset'
        self.data_save_path = "./dataset/val"
        self.dataset_make_mode = "mini_val"
        self.ckpt_dir = "./checkpoint/"
        self.log_dir = "./log"
        self.result_dir = "./result"