from monai.apps import download_and_extract
from pathlib import Path

def download_spleen_ds(resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar", md5 = "410d4a301da4e5b2f6f86ec3ddba524e"):
    data_dir = Path.cwd() / "datasets"
    print(data_dir / resource.split('/')[-1].split('.')[0])
    if not Path.exists(data_dir / resource.split('/')[-1].split('.')[0]):
        download_and_extract(resource, output_dir=data_dir, hash_val=md5)

if __name__ == '__main__':
    download_spleen_ds()