"""GCS upload utilities using google-cloud-storage SDK."""

from __future__ import annotations

import logging
import os
from typing import Sequence

logger = logging.getLogger(__name__)


def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """Parse ``gs://bucket/prefix`` into *(bucket_name, prefix)*.

    The returned *prefix* never has a leading ``/`` and always ends with
    ``/`` (unless the URI points at the bucket root, in which case it is
    the empty string).
    """
    path = gcs_uri.removeprefix("gs://")
    bucket, _, prefix = path.partition("/")
    prefix = prefix.strip("/")
    if prefix:
        prefix += "/"
    return bucket, prefix


def upload_files_to_gcs(
    local_dir: str,
    gcs_uri: str,
    extensions: Sequence[str],
) -> int:
    """Upload files matching *extensions* from *local_dir* to *gcs_uri*.

    Only regular files directly inside *local_dir* are considered (no
    recursion into sub-directories).

    Returns the number of files successfully uploaded.
    """
    from google.cloud import storage

    client = storage.Client()
    bucket_name, prefix = _parse_gcs_uri(gcs_uri)
    bucket = client.bucket(bucket_name)

    uploaded = 0
    for fname in os.listdir(local_dir):
        fpath = os.path.join(local_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if not fname.endswith(tuple(extensions)):
            continue
        blob = bucket.blob(prefix + fname)
        blob.upload_from_filename(fpath)
        uploaded += 1
        logger.debug("Uploaded %s -> gs://%s/%s", fname, bucket_name, blob.name)

    return uploaded


def sync_dir_to_gcs(local_dir: str, gcs_uri: str) -> int:
    """Upload all files in *local_dir* to *gcs_uri* (flat, no recursion).

    Returns the number of files uploaded.
    """
    from google.cloud import storage

    client = storage.Client()
    bucket_name, prefix = _parse_gcs_uri(gcs_uri)
    bucket = client.bucket(bucket_name)

    uploaded = 0
    for fname in os.listdir(local_dir):
        fpath = os.path.join(local_dir, fname)
        if not os.path.isfile(fpath):
            continue
        blob = bucket.blob(prefix + fname)
        blob.upload_from_filename(fpath)
        uploaded += 1
        logger.debug("Uploaded %s -> gs://%s/%s", fname, bucket_name, blob.name)

    return uploaded
