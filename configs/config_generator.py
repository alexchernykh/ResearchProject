def create_config_from_params(op_name: str,
                              path_location: str,
                              op_phase: str,
                              train: bool = False,
                              save_img_and_mask: bool = False,
                              classes_list=None) -> dict:
    """
    Generating config file that will be used to train/ evaluate the neural network.
    :param op_name: representing a 8 digit date stored as str f.e 20190917
    :param path_location: folder where data is located to feed it into the network f.e crossval_tme
    :param op_phase: name of the OP phase f.e TME
    :param train: training/validation
    :param save_img_and_mask: boolean value to either save the generated predicted masks and images or not
    :param classes_list: list of class labels
    :return:
        a created config file stored in a dict
    """
    if classes_list is None:
        classes_list = []
    created_config = {
        "OP_PHASE": op_phase,
        "OP_PATH_LOCATION": path_location,
        "OP_NAME": op_name,
        "NEW_DATA": True,
        "POINT_REND": False,
        "PLANE": False,
        "TRAIN": {"NUM_WORKERS": 2,
                  "BATCH": 2,
                  "BASE_LR": 0.00025,
                  "MAXITER": 400000,
                  "ROI_HEADS_BATCH_SIZE_PER_IMAGE": 128,
                  "OUTPUT_DIR": f'../detectron/output/{path_location}/{op_name}/'
                  },
        # validation data
        "VAL": {"NAME_MODEL_FILE": f"model_final_{op_phase}_{op_name}.pth",
                "SAVE_JSON": f"./../csv/detectron/{path_location}/predicted_plane/{op_name}.json",
                "DIR_TO_SAVE": f"{path_location}/{op_name}"
                },
        # data for training
        "NEW": {"JSON_DESCRIPTION": 'dataset_registration_detectron2_multilabeled_all_layers.json',
                "CLASSES_LIST": classes_list,
                "PATH_TO_DATA": f"../data/output/{path_location}/{op_name}/",
                "PATH_TO_TRAINING_DATA": f"../data/output/{path_location}/{op_name}/train/",
                "PATH_TO_VALIDATION_DATA": f"../data/output/{path_location}/{op_name}/val/"
                },

        # parameters to extract data from annotated files represented as NRRD and store it in masks + image folder
        # in addition
        "CREATE_DATASET": {"FOLDER_DATA_NRRD": "../data/nrrds_to_process/*.nrrd",
                           "IMG_DIR_TRAIN": f"../data/output/{path_location}/{op_name}/train/*.png",
                           "IMG_DIR_VAL": f"../data/output/{path_location}/{op_name}/val/*.png",
                           "PADDING": 2,
                           "OUTPUT_FILE": 'dataset_registration_detectron2_multilabeled_all_layers.json',
                           "OUTPUT_FILE_VAL": 'dataset_registration_detectron2_multilabeled_all_layers.json',
                           'SAVE_MASKS_LOCATION': f"../data/output/{path_location}/{op_name}/masks/",
                           "SAVE_IMAGES_LOCATION_TRAIN": f"../data/output/{path_location}/{op_name}/train/",
                           "SAVE_IMAGES_LOCATION_VAL": f"../data/output/{path_location}/{op_name}/val/",
                           'SAVE_MASKS_LOCATION_VAL': f"../data/output/{path_location}/{op_name}/masks_val/",
                           #  final_op_mode
                           "TRAIN": train,
                           "VERBOSE": True,
                           "SAVE_IMG_AND_MASKS": save_img_and_mask,
                           "CREATE_DESCRIPTION_FILE": True
                           }
    }
    return created_config
