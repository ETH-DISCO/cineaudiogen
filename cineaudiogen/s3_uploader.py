"""
S3 Uploader for Streaming Pipeline

Uploads generated scenes to S3 and deletes local files to save disk space.
"""

import os
import shutil
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed


class S3StreamingUploader:
    """Upload scenes to S3 and cleanup local files with optimized parallel transfers.

    Works with any S3-compatible object storage; credentials/endpoint come from the
    standard boto3 configuration (~/.aws/credentials, ~/.aws/config, or env vars).
    """

    def __init__(self, bucket_name: str, prefix: str = "cinematic_audio_scenes",
                 max_workers: int = 10):
        """
        Args:
            bucket_name: S3 bucket name
            prefix: S3 key prefix (folder path)
            max_workers: Number of parallel upload threads (default: 10)
        """
        # Lazy import so the package works without boto3 unless S3 is used
        # (install with `pip install cineaudiogen[s3]`).
        import boto3
        from boto3.s3.transfer import TransferConfig

        self.bucket_name = bucket_name
        self.prefix = prefix
        self.max_workers = max_workers
        self.s3_client = boto3.client('s3')

        # Optimized transfer configuration
        # - Multipart uploads for files > 8 MB
        # - 10 MB chunk size (good for audio files)
        # - 10 concurrent threads per file for multipart
        self.transfer_config = TransferConfig(
            multipart_threshold=8 * 1024 * 1024,  # 8 MB
            multipart_chunksize=10 * 1024 * 1024,  # 10 MB chunks
            max_concurrency=10,  # Parallel chunks per file
            use_threads=True
        )

    def upload_scene(self, scene_dir: str, delete_local: bool = True) -> Dict[str, str]:
        """
        Upload a complete scene directory to S3.

        Args:
            scene_dir: Local path to scene directory (e.g., data/parallel_output/expr_xxx)
            delete_local: If True, delete local files after successful upload

        Returns:
            Dict with upload status and S3 paths
        """
        scene_name = os.path.basename(scene_dir)
        s3_prefix = f"{self.prefix}/{scene_name}"

        uploaded_files = []
        failed_files = []

        # Collect all files to upload
        files_to_upload = []
        for root, dirs, files in os.walk(scene_dir):
            for filename in files:
                local_path = os.path.join(root, filename)
                rel_path = os.path.relpath(local_path, scene_dir)
                s3_key = f"{s3_prefix}/{rel_path}"
                files_to_upload.append((local_path, s3_key))

        # Upload files in parallel using ThreadPoolExecutor
        def upload_single_file(local_path, s3_key):
            """Upload a single file with optimized transfer config."""
            try:
                self.s3_client.upload_file(
                    local_path,
                    self.bucket_name,
                    s3_key,
                    Config=self.transfer_config
                )
                return {
                    'success': True,
                    'local': local_path,
                    's3_key': s3_key,
                    'size_mb': os.path.getsize(local_path) / (1024 * 1024)
                }
            except Exception as e:
                return {
                    'success': False,
                    'local': local_path,
                    'error': str(e)
                }

        # Parallel upload with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(upload_single_file, lp, s3k): (lp, s3k)
                      for lp, s3k in files_to_upload}

            for future in as_completed(futures):
                result = future.result()
                if result['success']:
                    uploaded_files.append(result)
                else:
                    failed_files.append(result)

        # Only delete if all uploads succeeded
        if delete_local and not failed_files:
            try:
                shutil.rmtree(scene_dir)
                deleted = True
            except Exception as e:
                deleted = False
                print(f"  [S3 Upload] Warning: Could not delete {scene_dir}: {e}")
        else:
            deleted = False

        total_size_mb = sum(f['size_mb'] for f in uploaded_files)

        return {
            'success': len(failed_files) == 0,
            'scene_name': scene_name,
            's3_prefix': f"s3://{self.bucket_name}/{s3_prefix}",
            'uploaded_count': len(uploaded_files),
            'failed_count': len(failed_files),
            'total_size_mb': total_size_mb,
            'local_deleted': deleted,
            'failed_files': failed_files if failed_files else None
        }

    def list_uploaded_scenes(self) -> list:
        """List all scenes already uploaded to S3."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{self.prefix}/",
                Delimiter='/'
            )

            # Extract scene names from common prefixes
            scenes = []
            for prefix_obj in response.get('CommonPrefixes', []):
                prefix_path = prefix_obj['Prefix']
                scene_name = prefix_path.rstrip('/').split('/')[-1]
                scenes.append(scene_name)

            return scenes
        except Exception as e:
            print(f"  [S3 Upload] Error listing scenes: {e}")
            return []

    def download_scene(self, scene_name: str, local_output_dir: str) -> bool:
        """
        Download a scene from S3 to local directory.

        Args:
            scene_name: Scene name (e.g., expr_xxx)
            local_output_dir: Local directory to download to

        Returns:
            True if successful, False otherwise
        """
        s3_prefix = f"{self.prefix}/{scene_name}"
        scene_dir = os.path.join(local_output_dir, scene_name)
        os.makedirs(scene_dir, exist_ok=True)

        try:
            # List all objects with this prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix)

            for page in pages:
                for obj in page.get('Contents', []):
                    s3_key = obj['Key']

                    # Compute local path
                    rel_path = s3_key[len(s3_prefix)+1:]  # Remove prefix + '/'
                    local_path = os.path.join(scene_dir, rel_path)

                    # Create parent directories
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)

                    # Download file
                    self.s3_client.download_file(self.bucket_name, s3_key, local_path)

            return True
        except Exception as e:
            print(f"  [S3 Download] Error downloading {scene_name}: {e}")
            return False


def get_s3_uploader(bucket_name: Optional[str] = None) -> Optional[S3StreamingUploader]:
    """
    Get S3 uploader instance if configured.

    Args:
        bucket_name: S3 bucket name, or None to read from environment

    Returns:
        S3StreamingUploader instance or None if not configured
    """
    bucket = bucket_name or os.environ.get('S3_BUCKET_NAME')

    if not bucket:
        return None

    prefix = os.environ.get('S3_PREFIX', 'cinematic_audio_scenes')

    return S3StreamingUploader(bucket, prefix)
