import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

class S3Manager:
    def __init__(self, bucket_name, region="us-east-1"):
        """
        Initialize S3Manager with a bucket name and AWS region.

        Args:
            bucket_name (str): Name of the S3 bucket.
            region (str): AWS region where the bucket is located.
        """
        self.bucket_name = bucket_name
        self.s3 = boto3.client('s3', region_name=region)

    def upload_folder(self, local_folder, s3_prefix):
        """
        Upload a local folder to an S3 bucket.

        Args:
            local_folder (str): Path to the local folder.
            s3_prefix (str): Prefix (folder name) in S3 where files will be stored.

        Returns:
            str: Name of the folder in S3.
        """
        for root, dirs, files in os.walk(local_folder):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_folder)
                s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
                try:
                    print(f"Uploading {local_path} to s3://{self.bucket_name}/{s3_key}")
                    self.s3.upload_file(local_path, self.bucket_name, s3_key)
                except NoCredentialsError:
                    print("AWS credentials not found. Please configure them using 'aws configure'.")
                    raise
                except Exception as e:
                    print(f"Error uploading {local_path}: {e}")
                    raise
        return s3_prefix

    def download_folder(self, s3_prefix, local_folder):
        """
        Download all files from an S3 prefix to a local folder.

        Args:
            s3_prefix (str): Prefix (folder name) in S3 to download files from.
            local_folder (str): Path to the local folder to save downloaded files.

        Returns:
            str: Name of the local folder where files are saved.
        """
        os.makedirs(local_folder, exist_ok=True)
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=s3_prefix)
            if 'Contents' in response:
                for obj in response['Contents']:
                    s3_key = obj['Key']
                    relative_path = os.path.relpath(s3_key, s3_prefix)
                    local_file_path = os.path.join(local_folder, relative_path)
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    print(f"Downloading {s3_key} to {local_file_path}")
                    self.s3.download_file(self.bucket_name, s3_key, local_file_path)
            else:
                print(f"No files found in S3 with prefix '{s3_prefix}'")
        except ClientError as e:
            print(f"Error downloading files: {e}")
            raise
        return local_folder




if __name__ == "__main__":
    bucket_name = "your-bucket-name"
    local_folder = "/path/to/local/folder"
    s3_prefix = "uploads/folder_name"

    #Upload

    s3_manager = S3Manager(bucket_name)
    folder_in_s3 = s3_manager.upload_folder(local_folder, s3_prefix)
    print(f"Folder uploaded to S3: {folder_in_s3}")


    #Download

    s3_manager = S3Manager(bucket_name)
    downloaded_folder = s3_manager.download_folder(s3_prefix, local_folder)
    print(f"Folder downloaded to: {downloaded_folder}")
