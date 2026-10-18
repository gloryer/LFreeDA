#!/usr/bin/env python3
"""Download LFreeDA data directly from Hugging Face (Python >=3.8).

The default is the complete July-to-August artifact subset. Use --all for
the full dataset, or --smoke-test for only the August model and Step II outputs.
All modes use the pinned revision below.

Dependency: requests. By default no account/token is required. HF_TOKEN is
optional and is sent only to huggingface.co, never to a signed download host.
Existing regular files are skipped without checking their size or contents.
The progress bar counts completed and skipped files as a percentage of the total.
Checksum verification is an optional separate step using
artifact/artifact-SHA256SUMS for the artifact subset.
"""

import argparse
import concurrent.futures as futures
import email.utils
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import threading
import time
from urllib.parse import quote, urlsplit

import requests


REPO = "AdrianLI1/LFreeDA"
REVISION = "94ea25bbddc93d881b4beb526a8024d9be12a2f1"
HUB = "https://huggingface.co"
BATCH_URL = HUB + "/datasets/" + REPO + ".git/info/lfs/objects/batch"
PREFIXES = tuple(
    ["graph_features/mb24/" + m + "/" for m in ("March", "April", "May", "July", "Aug")]
    + ["image_features/mb24/" + m + "/" for m in ("march", "april", "may", "july", "aug")]
    + ["labels/mb24/" + m + "/" for m in ("March", "April", "May", "July", "Aug")]
    + ["graph_features/benign_source/", "graph_features/benign_target/",
       "image_features/benign_source/", "image_features/benign_target/",
       "stepI_trained_models/mb24/aug/", "stepII_constructed_datasets/mb24/aug/"]
)
SMOKE_PREFIXES = PREFIXES[-2:]
LOCAL = threading.local()
COOLDOWN_LOCK = threading.Lock()
COOLDOWN_UNTIL = 0.0
OUTPUT_LOCK = threading.Lock()
PROGRESS_LINE = ""


def log_message(message):
    """Keep retry messages readable while a terminal progress bar is visible."""
    with OUTPUT_LOCK:
        if PROGRESS_LINE:
            print("\r" + " " * len(PROGRESS_LINE) + "\r", end="", file=sys.stderr)
        print(message, file=sys.stderr, flush=True)
        if PROGRESS_LINE:
            print(PROGRESS_LINE, end="", file=sys.stderr, flush=True)


class DownloadProgress:
    """Display completed/skipped file counts without inspecting any files."""

    def __init__(self, total, completed):
        self.total = total
        self.terminal = sys.stderr.isatty()
        self.last_update = 0.0
        self.last_completed = None
        self.update(completed, force=True)

    def update(self, completed, force=False):
        global PROGRESS_LINE
        now = time.monotonic()
        interval = 1.0 if self.terminal else 30.0
        if not force and (completed == self.last_completed or
                          (completed < self.total and now - self.last_update < interval)):
            return
        # Round down so 100.0% appears only when every file is accounted for.
        percent = (completed * 1000 // self.total) / 10.0
        filled = completed * 25 // self.total
        line = "Files [%s%s] %5.1f%%  %s/%s" % (
            "#" * filled, "-" * (25 - filled), percent, completed, self.total)
        with OUTPUT_LOCK:
            if self.terminal:
                print("\r" + line.ljust(len(PROGRESS_LINE)), end="", file=sys.stderr, flush=True)
                PROGRESS_LINE = line
            else:
                print(line, file=sys.stderr, flush=True)
        self.last_update = now
        self.last_completed = completed

    def close(self):
        global PROGRESS_LINE
        with OUTPUT_LOCK:
            if self.terminal and PROGRESS_LINE:
                print(file=sys.stderr, flush=True)
                PROGRESS_LINE = ""


class NoImplicitAuth(requests.auth.AuthBase):
    def __call__(self, request):
        return request


class SafeSession(requests.Session):
    def send(self, request, **kwargs):
        if urlsplit(request.url).scheme != "https":
            raise RuntimeError("Refusing a non-HTTPS redirect")
        return super().send(request, **kwargs)

    def rebuild_auth(self, prepared, response):
        # Keep proxy/environment support without implicitly reading .netrc.
        if self.should_strip_auth(response.request.url, prepared.url):
            prepared.headers.pop("Authorization", None)


def session():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = SafeSession()
        LOCAL.session.auth = NoImplicitAuth()
        LOCAL.session.headers["User-Agent"] = "LFreeDA-AE-downloader/1.0"
    return LOCAL.session


def retry_delay(response, attempt):
    value = response.headers.get("Retry-After", "")
    try:
        return max(1.0, float(value))
    except ValueError:
        try:
            return max(1.0, email.utils.parsedate_to_datetime(value).timestamp() - time.time())
        except (ValueError, TypeError):
            pass
    match = re.search(r"\bt=(\d+)", response.headers.get("RateLimit", ""))
    if match:
        return float(match.group(1)) + 1
    # Hub limits use five-minute windows when no reset header is available.
    return 301 if response.status_code == 429 else min(2 ** attempt, 60)


def request(method, url, headers=None, **kwargs):
    global COOLDOWN_UNTIL
    if urlsplit(url).scheme != "https":
        raise RuntimeError("Refusing a non-HTTPS download URL")
    headers = dict(headers or {})
    token = os.environ.get("HF_TOKEN")
    if token and urlsplit(url).hostname == "huggingface.co":
        headers.setdefault("Authorization", "Bearer " + token)
    for attempt in range(6):
        with COOLDOWN_LOCK:
            delay = COOLDOWN_UNTIL - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        try:
            response = session().request(method, url, headers=headers,
                                         timeout=(30, 120), **kwargs)
        except requests.RequestException as error:
            if attempt == 5:
                raise RuntimeError("Network retries exhausted (%s)" % type(error).__name__) from None
            time.sleep(min(2 ** attempt, 60))
            continue
        if response.status_code == 429 or 500 <= response.status_code < 600:
            delay = retry_delay(response, attempt)
            status = response.status_code
            response.close()
            if attempt == 5:
                raise RuntimeError("HTTP %s retries exhausted; rerun later to resume" % status)
            log_message("HTTP %s from %s; retrying after %.0fs" %
                        (status, urlsplit(url).hostname, delay))
            with COOLDOWN_LOCK:
                COOLDOWN_UNTIL = max(COOLDOWN_UNTIL, time.monotonic() + delay)
            continue
        if response.status_code != 200:
            status = response.status_code
            response.close()
            raise RuntimeError("HTTP %s from %s" % (status, urlsplit(url).hostname))
        return response
    raise RuntimeError("Request retries exhausted")


def checked_path(dest, name):
    relative = PurePosixPath(name)
    if (relative.is_absolute() or not relative.parts or ".." in relative.parts
            or "\\" in name or any(ord(char) < 32 for char in name)):
        raise RuntimeError("Unsafe repository path")
    return dest.joinpath(*relative.parts)


def prepare_destination(dest, files):
    """Inventory each required directory once, without reading file sizes/data."""
    root = {"dirs": {}, "files": set()}
    for file in files:
        parts = PurePosixPath(file["rfilename"]).parts
        node = root
        for name in parts[:-1]:
            node = node["dirs"].setdefault(name, {"dirs": {}, "files": set()})
        node["files"].add(parts[-1])

    # Inspect destination ancestors once; do not resolve away a symlink.
    for path in list(reversed(dest.parents)) + [dest]:
        try:
            info = path.lstat()
        except FileNotFoundError:
            path.mkdir()
        else:
            if not stat.S_ISDIR(info.st_mode):
                raise RuntimeError("Refusing symlink or non-directory: " + str(path))

    existing = set()

    def inventory(path, node, prefix, new=False):
        if node["files"].intersection(node["dirs"]):
            raise RuntimeError("Conflicting repository paths")
        found_dirs = set()
        if not new:
            with os.scandir(str(path)) as entries:
                for entry in entries:
                    if entry.name in node["files"]:
                        if not entry.is_file(follow_symlinks=False):
                            raise RuntimeError("Refusing symlink or non-regular file: " + entry.path)
                        existing.add(prefix + entry.name)
                    elif entry.name in node["dirs"]:
                        if not entry.is_dir(follow_symlinks=False):
                            raise RuntimeError("Refusing symlink or non-directory: " + entry.path)
                        found_dirs.add(entry.name)
        for name, child in node["dirs"].items():
            child_path = path / name
            created = False
            if name not in found_dirs:
                try:
                    child_path.mkdir()
                    created = True
                except FileExistsError:
                    # A concurrent creator must not turn a required directory
                    # into a symlink or another kind of filesystem entry.
                    if not stat.S_ISDIR(child_path.lstat().st_mode):
                        raise RuntimeError("Refusing symlink or non-directory: " + str(child_path))
            inventory(child_path, child, prefix + name + "/", new=created)

    inventory(dest, root, "")
    return existing


def temporary(path):
    fd, name = tempfile.mkstemp(prefix=".lfreeda-", suffix=".part", dir=str(path.parent))
    os.close(fd)
    return Path(name)


def publish(part, path):
    # An atomic hard link publishes the complete temporary file without ever
    # replacing a destination created since the inventory. The temporary link
    # is removed by the caller, so downloaded files have independent inodes.
    try:
        os.link(str(part), str(path))
    except FileExistsError:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise RuntimeError("Refusing symlink or non-regular file: " + str(path))


def download_group(files, action, dest):
    paths = [checked_path(dest, file["rfilename"]) for file in files]
    source = temporary(paths[0])
    try:
        for attempt in range(3):
            try:
                received = 0
                with request("GET", action["href"], headers=action.get("header", {}),
                             stream=True) as response, source.open("wb") as output:
                    for block in response.iter_content(1024 * 1024):
                        output.write(block)
                        received += len(block)
                if received != files[0]["size"]:
                    raise RuntimeError("Download size mismatch: %s (expected %s bytes, received %s)" %
                                       (paths[0], files[0]["size"], received))
                break
            except requests.RequestException as error:
                if attempt == 2:
                    raise RuntimeError("Download interrupted (%s); rerun to resume" %
                                       type(error).__name__) from None
                time.sleep(2 ** attempt)
        publish(source, paths[0])
        for path in paths[1:]:
            part = temporary(path)
            try:
                # Copy only the freshly downloaded temporary file, never an
                # existing destination whose contents have not been checked.
                shutil.copyfile(str(source), str(part))
                publish(part, path)
            finally:
                part.unlink()
    finally:
        source.unlink()
    return len(files)


def batch_actions(groups):
    objects = [{"oid": files[0]["lfs"]["sha256"], "size": files[0]["size"]}
               for files in groups]
    with request("POST", BATCH_URL,
                 headers={"Accept": "application/vnd.git-lfs+json",
                          "Content-Type": "application/vnd.git-lfs+json"},
                 json={"operation": "download", "transfers": ["basic"],
                       "objects": objects}) as response:
        result = response.json()
    wanted = {obj["oid"]: obj["size"] for obj in objects}
    actions = {}
    for obj in result["objects"]:
        if obj["oid"] not in wanted or obj.get("size") != wanted[obj["oid"]]:
            raise RuntimeError("Unexpected LFS batch object")
        if "error" in obj or "download" not in obj.get("actions", {}):
            raise RuntimeError("LFS batch could not resolve object " + obj["oid"])
        actions[obj["oid"]] = obj["actions"]["download"]
    if set(actions) != set(wanted):
        raise RuntimeError("Incomplete LFS batch response")
    return actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", required=True, type=Path, help="New dataset directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="Retrieve the entire repository")
    mode.add_argument("--smoke-test", action="store_true",
                      help="Only August Step I model and Step II outputs (not the complete artifact)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel downloads (default: 4)")
    parser.add_argument("--dry-run", action="store_true", help="Inspect remote metadata without accessing the destination")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be between 1 and 32")
    dest = Path(os.path.abspath(str(args.dest.expanduser())))
    with request("GET", HUB + "/api/datasets/" + REPO + "/revision/" + REVISION,
                 params={"blobs": "true"}) as response:
        metadata = response.json()
    if metadata["sha"] != REVISION:
        raise RuntimeError("Unexpected repository revision")
    prefixes = SMOKE_PREFIXES if args.smoke_test else PREFIXES
    files = sorted((file for file in metadata["siblings"]
                    if args.all or file["rfilename"].startswith(prefixes)),
                   key=lambda file: file["rfilename"])
    if not files:
        raise RuntimeError("No matching files")
    for file in files:
        checked_path(dest, file["rfilename"])
    label = "full repository" if args.all else "model/output smoke test" if args.smoke_test else "artifact subset"
    print("%s @ %s\n%s: %s files, %s bytes (%.2f GiB)" %
          (REPO, REVISION, label, len(files), sum(file["size"] for file in files),
           sum(file["size"] for file in files) / 1024 ** 3), flush=True)
    if args.dry_run:
        return
    existing = prepare_destination(dest, files)
    groups = {}
    small_files = []
    for file in files:
        if file["rfilename"] in existing:
            continue
        if "lfs" in file:
            groups.setdefault(file["lfs"]["sha256"], []).append(file)
        else:
            small_files.append(file)
    pending = list(groups.values())
    completed = len(existing)

    progress = DownloadProgress(len(files), completed)
    try:
        with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for offset in range(0, len(pending), 500):
                batch = pending[offset:offset + 500]
                jobs = set()
                if batch:
                    actions = batch_actions(batch)
                    jobs = {pool.submit(download_group, group,
                                        actions[group[0]["lfs"]["sha256"]], dest) for group in batch}
                while jobs:
                    done, jobs = futures.wait(jobs, timeout=30, return_when=futures.FIRST_COMPLETED)
                    for job in done:
                        completed += job.result()
                    progress.update(completed)
            for file in small_files:
                action = {"href": HUB + "/datasets/" + REPO + "/resolve/" + REVISION
                          + "/" + quote(file["rfilename"], safe="/")}
                completed += download_group([file], action, dest)
                progress.update(completed)
    finally:
        progress.close()
    print("Complete: %s files in %s" % (completed, dest), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        print("Error: " + str(error), file=sys.stderr)
        sys.exit(1)
