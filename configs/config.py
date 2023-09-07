import os

from datetime import datetime

import utils.logging as logging


class Config(object):
    def __init__(self, config_yaml):
        self.max_depth_eval = None

        self.config_yaml = config_yaml
        self.user_name = config_yaml['USER_NAME']
        self.setting_for_system()
        self.settings_for_dataset()
        self.setting_for_path()
        self.settings_for_backbone()
        # self.settings_for_decoder()
        self.settings_for_training()
        self.settings_for_evaluation()
        self.settings_for_etc()

    def setting_for_system(self):
        self.gpu_or_cpu = 'gpu'
        self.root = os.path.dirname(os.path.abspath(__file__)) + "/../"
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config_yaml['GPU_ID']

    def settings_for_dataset(self):
        self.dataset    = self.config_yaml['DATASET_NAME']
        self.data_dir   = self.config_yaml['DATA_PATH']

        self.batch_size = self.config_yaml['BATCH_SIZE']
        self.workers    = self.config_yaml['WORKERS']
        self.crop_h     = self.config_yaml['CROP_HEIGHT']
        self.crop_w     = self.config_yaml['CROP_WIDTH']
        self.max_depth  = self.config_yaml['MAX_DEPTH']

        self.image_interval_range = self.config_yaml['IMAGE_INTERVAL_RANGE']

    def setting_for_path(self):
        self.description = '_'.join([self.user_name, self.dataset])
        self.proj_dir    = os.path.dirname(os.path.abspath(__file__)) + "/../"
        self.exp_name    = datetime.now().strftime('%m%d_%H%M%S')
        self.log_dir     = os.path.join(self.proj_dir, 'logs', self.description + '_' + self.exp_name)

    def settings_for_backbone(self):
        self.model_scale        = self.config_yaml['MODEL_SCALE']
        self.backbone           = self.config_yaml['BACKBONE']
        self.decoder            = self.config_yaml['DECODER']
        self.pretrained         = self.config_yaml['SWIN']['PRETRAINED_SWIN']
        self.use_checkpoint     = self.config_yaml['SWIN']['USE_CHECKPOINT_SWIN']
        if "cnn_transformer" in self.backbone:
            self.cnn_model      = self.config_yaml['CNN_TRANSFORMER']['CNN_MODEL']
        elif "resnet_only" in self.backbone:
            self.cnn_model      = self.config_yaml['RESNET_ONLY']['CNN_MODEL']
        else:
            self.cnn_model      = ""
        self.transformer_ff_dim = self.config_yaml['CNN_TRANSFORMER']['TRANSFORMER_FF_DIM']

        self.depths               = self.config_yaml['SWIN']['DEPTHS']
        self.window_size          = self.config_yaml['SWIN']['WINDOW_SIZE']
        self.pretrain_window_size = self.config_yaml['SWIN']['PRETRAIN_WINDOW_SIZE']
        self.use_shift            = self.config_yaml['SWIN']['USE_SHIFT']
        self.shift_window_test    = self.config_yaml['SWIN']['SHIFT_WINDOW_TEST']
        self.shift_size           = self.config_yaml['SWIN']['SHIFT_SIZE']
        self.drop_path_rate       = self.config_yaml['SWIN']['DROP_PATH_RATE']

    # def settings_for_decoder(self):
    #     self.num_deconv = self.config_yaml['NUM_DECONV']
    #     self.num_filters = self.config_yaml['NUM_FILTERS']
    #     self.deconv_kernels = self.config_yaml['DECONV_KERNELS']

    def settings_for_training(self):
        self.epochs       = self.config_yaml['EPOCH']
        self.max_lr       = self.config_yaml['MAX_LEARNING_RATE']
        self.min_lr       = self.config_yaml['MIN_LEARNING_RATE']
        self.weight_decay = self.config_yaml['WEIGHT_DECAY']
        self.layer_decay  = self.config_yaml['LAYER_DECAY']

        self.pro_bar    = self.config_yaml['PRO_BAR']
        self.val_freq   = self.config_yaml['VALIDATION_FREQUENCY']
        self.save_freq  = self.config_yaml['SAVE_FREQUENCY']
        self.print_freq = self.config_yaml['PRINT_FREQUENCY']

        self.resume_from = self.config_yaml['RESUME_FROM']
        #self.auto_resume = self.config_yaml['AUTO_RESUME']
        self.save_model  = self.config_yaml['SAVE_MODEL']
        self.save_result = self.config_yaml['SAVE_RESULT']

        self.loss_lambda1 = self.config_yaml['LOSS_LAMBDA1']
        self.loss_lambda2 = self.config_yaml['LOSS_LAMBDA2']

    def settings_for_evaluation(self):
        self.max_depth_eval = self.config_yaml['MAX_DEPTH_EVAL']
        self.min_depth_eval = self.config_yaml['MIN_DEPTH_EVAL']
        self.do_kb_crop     = self.config_yaml['DO_KB_CROP']
        self.flip_test      = self.config_yaml['FLIP_TEST']

        self.save_eval_pngs = self.config_yaml['SAVE_EVAL_PNGS']
        self.save_visualize = self.config_yaml['SAVE_VISUALIZE']
        self.do_evaluate    = self.config_yaml['DO_EVALUATE']
        self.ckpt_dir       = self.config_yaml['CHECKPOINT_DIR']

    def settings_for_etc(self):
        
        self.datetime = datetime.now().strftime('%m%d')

    def settings_for_test_void_with_custom_network(self):
        pretrain = self.pretrained.split('.')[0]
        maxlrstr = str(self.max_lr).replace('.', '')
        minlrstr = str(self.min_lr).replace('.', '')
        layer_decaystr = str(self.layer_decay).replace('.', '')
        weight_decaystr = str(self.weight_decay).replace('.', '')
        num_filter = str(self.num_filters[0]) if self.num_deconv > 0 else ''
        num_kernel = str(self.deconv_kernels[0]) if self.num_deconv > 0 else ''
        name = [str(self.batch_size), pretrain.split('/')[-1], 'deconv' + str(self.num_deconv),
                str(num_filter), str(num_kernel), str(self.crop_h), str(self.crop_w), maxlrstr, minlrstr,
                layer_decaystr, weight_decaystr, str(self.epochs)]
        if 'swin' in self.backbone:
            for i in self.window_size:
                name.append(str(i))
            for i in self.depths:
                name.append(str(i))

        exp_name = '_'.join(name) + '_downscale16'
        print('This experiments: ', exp_name)

        # Logging
        self.exp_name = exp_name
        self.result_dir = os.path.join(self.proj_dir, 'results', self.description + '_' + self.exp_name)
        os.makedirs(self.result_dir, exist_ok=True)
