import tensorflow as tf

from data_loader.automap_inference_data_generator import InferenceDataGenerator
from trainers.automap_inferencer import AUTOMAP_Inferencer
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
    data = InferenceDataGenerator(config)
    
    model = tf.keras.models.load_model(config.loadmodel_dir)
    
    inferencer = AUTOMAP_Inferencer(model, data, config)
    inferencer.inference()


if __name__ == '__main__':
    main()