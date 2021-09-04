"""
"""
from io import open
import json
import logging
from functools import wraps
from hashlib import sha256
from pathlib import Path
import os
import shutil
import sys
import tempfile
from urllib.parse import urlparse

import boto3
import requests
from botocore.exceptions import ClientError
from tqdm import tqdm
from tensorboardX import SummaryWriter
from time import gmtime, strftime
from bisect import bisect

import pdb

PYTORCH_PRETRAINED_BERT_CACHE = Path(
    os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", Path.home() / ".pytorch_pretrained_bert")
)


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def lr_warmup(step, ):
    if (
        cfg["training_parameters"]["use_warmup"] is True
        and i_iter <= cfg["training_parameters"]["warmup_iterations"]
    ):
        alpha = float(i_iter) / float(cfg["training_parameters"]["warmup_iterations"])
        return cfg["training_parameters"]["warmup_factor"] * (1.0 - alpha) + alpha
    else:
        idx = bisect(cfg["training_parameters"]["lr_steps"], i_iter)
        return pow(cfg["training_parameters"]["lr_ratio"], idx)

class tbLogger(object):
    def __init__(self, log_dir, txt_dir, task_names, task_ids, task_num_iters, gradient_accumulation_steps, save_logger=True, txt_name='out.txt'):
        logger.info("logging file at: " + log_dir)

        self.save_logger=save_logger
        if self.save_logger:
            self.logger = SummaryWriter(log_dir=log_dir)

        self.txt_f = open(txt_dir + '/' + txt_name, 'w')
        self.task_id2name = {ids:name.replace('+', 'plus') for ids, name in zip(task_ids, task_names)}
        self.task_ids = task_ids
        self.task_loss = {task_id:0 for task_id in task_ids}
        self.task_loss_tmp = {task_id:0 for task_id in task_ids}
        self.task_score_tmp = {task_id:0 for task_id in task_ids}
        self.task_norm_tmp = {task_id:0 for task_id in task_ids}
        self.task_step = {task_id:0 for task_id in task_ids}
        self.task_step_tmp = {task_id:0 for task_id in task_ids}
        self.task_num_iters = task_num_iters
        self.epochId = 0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.task_loss_val = {task_id:0 for task_id in task_ids}
        self.task_score_val = {task_id:0 for task_id in task_ids}
        self.task_step_val = {task_id:0 for task_id in task_ids}
        self.task_datasize_val = {task_id:0 for task_id in task_ids}

    def txt_close(self):
        self.txt_f.close()

    def linePlot(self, step, val, split, key, xlabel="None"):
        if self.save_logger:
            self.logger.add_scalar(split + "/" + key, val, step)

    def step_train(self, epochId, stepId, loss, score, norm, task_id, split):
        self.task_loss[task_id] += loss
        self.task_loss_tmp[task_id] += loss
        self.task_score_tmp[task_id] += score
        self.task_norm_tmp[task_id] += norm
        self.task_step[task_id] += self.gradient_accumulation_steps
        self.task_step_tmp[task_id] += self.gradient_accumulation_steps
        self.epochId = epochId

        # plot on tensorboard.
        self.linePlot(stepId, loss, split, self.task_id2name[task_id] + '_loss')
        self.linePlot(stepId, score, split, self.task_id2name[task_id] + '_score')

    def step_val(self, epochId, loss, score, task_id, batch_size, split):
        self.task_loss_val[task_id] += loss
        self.task_score_val[task_id] += score
        self.task_step_val[task_id] += self.gradient_accumulation_steps
        self.task_datasize_val[task_id] += batch_size

    def showLossVal(self):
        progressInfo = "Eval Ep: %d " %self.epochId
        lossInfo = 'Validation '
        ave_score = 0
        ave_loss = 0
        for task_id in self.task_ids:
            loss = self.task_loss_val[task_id] / float(self.task_step_val[task_id])
            score = self.task_score_val[task_id] / float(self.task_datasize_val[task_id])
            ave_score += score
            ave_loss += loss
            lossInfo += '[%s]: loss %.3f score %.3f ' %(self.task_id2name[task_id], loss, score * 100.0)

            self.linePlot(self.epochId, loss, 'val', self.task_id2name[task_id] + '_loss')
            self.linePlot(self.epochId, score, 'val', self.task_id2name[task_id] + '_score')

        ave_score = ave_score / len(self.task_ids)
        self.task_loss_val = {task_id:0 for task_id in self.task_loss_val}
        self.task_score_val = {task_id:0 for task_id in self.task_score_val}
        self.task_datasize_val = {task_id:0 for task_id in self.task_datasize_val}
        self.task_step_val = {task_id:0 for task_id in self.task_ids}
        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)
        return ave_score

    def showLossTrain(self):
        # show the current loss, once showed, reset the loss. 
        lossInfo = ''
        for task_id in self.task_ids:
            if self.task_num_iters[task_id] > 0:
                if self.task_step_tmp[task_id]:
                    lossInfo += '[%s]: iter %d Ep: %.2f loss %.3f score %.3f lr %.6g ' %(self.task_id2name[task_id], \
                        self.task_step[task_id], self.task_step[task_id] / float(self.task_num_iters[task_id]), \
                                            self.task_loss_tmp[task_id] / float(self.task_step_tmp[task_id]), \
                                            self.task_score_tmp[task_id] / float(self.task_step_tmp[task_id]), \
                                            self.task_norm_tmp[task_id] / float(self.task_step_tmp[task_id]))
        
        logger.info(lossInfo)
        print(lossInfo, file=self.txt_f)

        self.task_step_tmp = {task_id:0 for task_id in self.task_ids}
        self.task_loss_tmp = {task_id:0 for task_id in self.task_ids}
        self.task_score_tmp =  {task_id:0 for task_id in self.task_ids}
        self.task_norm_tmp = {task_id:0 for task_id in self.task_ids}

def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode("utf-8")
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    return filename


def filename_to_url(filename, cache_dir=None):
    """
    Return the url and etag (which may be ``None``) stored for `filename`.
    Raise ``EnvironmentError`` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise EnvironmentError("file {} not found".format(cache_path))

    meta_path = cache_path + ".json"
    if not os.path.exists(meta_path):
        raise EnvironmentError("file {} not found".format(meta_path))

    with open(meta_path, encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
    url = metadata["url"]
    etag = metadata["etag"]

    return url, etag


def cached_path(url_or_filename, cache_dir=None):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if sys.version_info[0] == 3 and isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https", "s3"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))

def split_s3_path(url):
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad s3 path {}".format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    # Remove '/' at beginning of path.
    if s3_path.startswith("/"):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


def s3_request(func):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                raise EnvironmentError("file {} not found".format(url))
            else:
                raise
    return wrapper

@s3_request
def s3_etag(url):
    """Check ETag on S3 object."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag

@s3_request
def s3_get(url, temp_file):
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3")
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)

def http_get(url, temp_file):
    req = requests.get(url, stream=True)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def get_from_cache(url, cache_dir=None):
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Get eTag to add to filename, if it exists.
    if url.startswith("s3://"):
        etag = s3_etag(url)
    else:
        response = requests.head(url, allow_redirects=True)
        if response.status_code != 200:
            raise IOError(
                "HEAD request failed for url {} with status code {}".format(
                    url, response.status_code
                )
            )
        etag = response.headers.get("ETag")

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    if not os.path.exists(cache_path):
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info("%s not found in cache, downloading to %s", url, temp_file.name)

            # GET file object
            if url.startswith("s3://"):
                s3_get(url, temp_file)
            else:
                http_get(url, temp_file)

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)

            logger.info("copying %s to cache at %s", temp_file.name, cache_path)
            with open(cache_path, "wb") as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            logger.info("creating metadata file for %s", cache_path)
            meta = {"url": url, "etag": etag}
            meta_path = cache_path + ".json"
            with open(meta_path, "w", encoding="utf-8") as meta_file:
                json.dump(meta, meta_file)

            logger.info("removing temp file %s", temp_file.name)

    return cache_path


def read_set_from_file(filename):
    """
    Extract a de-duped collection (set) of text from a file.
    Expected file format is one item per line.
    """
    collection = set()
    with open(filename, "r", encoding="utf-8") as file_:
        for line in file_:
            collection.add(line.rstrip())
    return collection

def get_file_extension(path, dot=True, lower=True):
    ext = os.path.splitext(path)[1]
    ext = ext if dot else ext[1:]
    return ext.lower() if lower else ext

