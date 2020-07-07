import tensorflow as tf

from data_loader.automap_data_generator import DataGenerator
from models.automap_model import AUTOMAP_Basic_Model 
from trainers.automap_trainer import AUTOMAP_Trainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args


def main():
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    create_dirs([config.summary_dir, config.checkpoint_dir])
    data = DataGenerator(config)
    
    model = AUTOMAP_Basic_Model(config)
    trainer = AUTOMAP_Trainer(model, data, config)
    trainer.train()


if __name__ == '__main__':
    main()
