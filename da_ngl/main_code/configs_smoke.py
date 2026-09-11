"""Tiny smoke-test config (few samples, few iterations) - delete if unused."""
from configs import *  # noqa: F401,F403

work_dir = '/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/linan/linan_dev/gnss/da_ngl/main_code/work_dir_smoke'

dates_train_range = ['2022010100', '2022010218']
dates_val_range = ['2022010300', '2022010318']
dates_test_range = ['2022010400', '2022010418']

num_iteration = 4
num_epochs = 2
batch_size = 2
num_workers = 2
prefetch_factor = 2
model_embed_dim = 32
model_depth = (1, 1, 1)
