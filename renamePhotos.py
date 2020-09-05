import os
from PIL import Image
import argparse

from_path = "./data/images"
to_path = "./data/images"
starting_num = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--delete", help="Delete the original photos", action="store_true")
    args = parser.parse_args()


    files = os.listdir(from_path)
    files.sort()

    cnt = starting_num
    for file in files:
        ext = os.path.splitext(file)[1]
        if file[0] != "." and (ext.lower() in [".png", ".jpeg", ".jpg"]):
            new_name = "{:08d}".format(cnt) + ext.lower()

            image = Image.open(os.path.join(from_path, file))
            if args.delete:
                os.remove(os.path.join(from_path, file))
            image.save(os.path.join(to_path, new_name), ext[1:])

            print("Convert", file, "to", new_name)
            if args.delete:
                print(file, "DELETED")
            cnt += 1