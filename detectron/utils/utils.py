import os

path = os.getcwd()


def create_dir_strucutre(op_type: str, op_name: str) -> None:
    this_path = f"/Users/chernykh_alexander/Github/chirurgie_research/"

    # base_dir = f"{this_path}data/output/{op_type}"
    base_dir = f"{this_path}detectron/output/{op_type}"
    # {path_location}/{op_name}/val/*.png
    try:
        # os.mkdir(f"{base_dir}/{op_name}/")
        os.mkdir(f"{base_dir}/{op_name}/img")
        os.mkdir(f"{base_dir}/{op_name}/images_predicted_multilabeled")
        os.mkdir(f"{base_dir}/{op_name}/images_predicted_multilabeled/full_prediction")
        os.mkdir(f"{base_dir}/{op_name}/images_predicted_multilabeled/layerwise")
        for i in range(0, 12):
            os.mkdir(
                f"{base_dir}/{op_name}/images_predicted_multilabeled/layerwise_masks/{i}"
            )
        os.mkdir(f"{base_dir}/{op_name}/images_predicted_multilabeled/layerwise_masks")
        for i in range(0, 12):
            os.mkdir(
                f"{base_dir}/{op_name}/images_predicted_multilabeled/layerwise_masks/{i}"
            )
        # os.mkdir(f"{base_dir}/{op_name}/masks")
        # os.mkdir(f"{base_dir}/{op_name}/masks_val")
        # os.mkdir(f"{base_dir}/{op_name}/train")
        # os.mkdir(f"{base_dir}/{op_name}/val")
    except OSError:
        print("Creation of the directory %s failed" % f"{base_dir}/{op_name}")
    else:
        print("Successfully created the directory %s " % base_dir)


# # names = os.listdir("/Users/chernykh_alexander/Github/chirurgie_research/data/nrrds_to_process/")
# names = ["20200622"]
# op_type = "5phase"
# for name in names:
#     if name != '.DS_Store':
#         print(name)
#         create_dir_strucutre(op_name=name, op_type=op_type)


import os

src_l = "../../data/output/5phase/20200630/masks_val"
"../data/output/2phase/20200622/val/00001810_FIRST_INCISION_00000.png"
# op_pnase = "FIRST_INCISION"
index_nulls = 8
for count, filename in enumerate(os.listdir(src_l)):
    print(filename)
    print(count)
    # dst = "Hostel" + str(count) + ".png"
    src = f"{src_l}/{filename}"
    print(src)

    value = 11000 + count
    index = str(value).zfill(index_nulls)
    dst = f"{str(index)}_00.png"
    print(dst)

    # rename() function will
    # rename all the files
    os.rename(src, dst)
# os.rename('a.txt', 'b.kml')
