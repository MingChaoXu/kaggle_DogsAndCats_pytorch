# coding:utf8
import warnings


class DefaultConfig(object):
    env = 'default'  # visdom环境
    model = 'AlexNet'  # 使用的模型，名字必须与mdoels/__init__.py中的名字一致

    train_data_root = './data/train/'  # 训练集存放路径
    test_data_root = './data/test/'  # 测试集存放路径
    load_model_path = None # 'checkpoints/model.pth'  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 128  # batch size
    use_gpu = False #True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file):enter ipdb
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.1  # initial learning rate
    lr_decay = 0.95  # when val_loss increase lr = lr*lr_decay
    weight_decay = 1e-4  # 损失函数

    def parse(self, kwargs):
        '''
        根据字典kwargs更新config参数
        :param kwargs:
        :return:
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn('Warning: opt has not attribute %s' % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))

# DefaultConfig.parse = parse
opt = DefaultConfig()