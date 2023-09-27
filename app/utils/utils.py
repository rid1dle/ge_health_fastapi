import os
import glob
from pathlib import Path
from typing import Union

import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

BUCKET_NAME = "gehealthcare-04.appspot.com"
ENV_PATH = f"{Path(__file__).parent.parent}/env/google-firebase.json"


def verify_google(bucket_name):
    cred = credentials.Certificate(ENV_PATH)
    try:
        firebase_admin.initialize_app(cred, {"storageBucket": bucket_name})

    except BaseException:
        return True
    except Exception as e:
        raise f"Google Firebase connection failed due to {e}"

    return True


def download_from_google(
    bucker_path, file_path, bucket_name=BUCKET_NAME, overwrite=False
):
    verify_google(bucket_name)
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix=bucker_path)
    files = []
    for blob in blobs:
        filename = blob.name.replace("/", "_")
        if not os.path.exists(f"{file_path}/{filename}") or overwrite is True:
            files.append(filename)
            blob.download_to_filename(f"{file_path}/{filename}")
    return files


def clean_previous_pdfs(path_to_pdfs, list_of_pdfs: Union[None, list[str]] = None):
    files = glob.glob(f"{path_to_pdfs}/*.pdf")
    cleaned_files = []
    for f in files:
        file_name = f.split("/")[-1]
        if list_of_pdfs is not None:
            if str(file_name) in list_of_pdfs:
                cleaned_files.append(file_name)
                os.remove(f)
        else:
            cleaned_files.append(file_name)
            os.remove(f)

    return cleaned_files


def get_list_of_pdfs(path_to_pdfs):
    path_to_pdfs += "/*.pdf"
    files = glob.glob(path_to_pdfs)
    names = [os.path.basename(f) for f in files]
    return names
