import os
import gzip
import json
import openai
import jsonlines

from typing import List
import tempfile
import sys
from typing import List, Optional, Iterable

def make_printv(verbose: bool):
    def print_v(*args, **kwargs):
        if verbose:
            kwargs["flush"] = True
            print(*args, **kwargs)
        else:
            pass
    return print_v


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File `{path}` does not exist.")
    elif not path.endswith(".jsonl"):
        raise ValueError(f"File `{path}` is not a jsonl file.")
    items = []
    with jsonlines.open(path) as reader:
        for item in reader:
            items += [item]
    return items


def read_jsonl_map(path: str, primary_key='task_id') -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File `{path}` does not exist.")
    elif not path.endswith(".jsonl"):
        raise ValueError(f"File `{path}` is not a jsonl file.")
    items = {}
    with jsonlines.open(path) as reader:
        for item in reader:
            items[item[primary_key]] = item
    return items



def write_jsonl(path: str,
                data: List[dict],
                append: bool = False,
                key: Optional[str] = None,
                accum_fields: Iterable[str] = ("prompt_tokens", "completion_tokens", "cost"),
                stage2: bool = False) -> None:
    """
    Write a list of dicts to a JSONL file, with optional upsert-by-key and accumulation.

    Behavior:
      - If key is None:
          * append=False -> overwrite file (original behavior)
          * append=True  -> append new lines (original behavior)
      - If key is not None (upsert mode):
          * Read existing file (if present).
          * For each incoming item that has `item[key]`:
              - If a record with the same key exists, replace it, but:
                  · For any field in `accum_fields` present in either record,
                    set updated[field] = old[field] + new[field] (numeric add).
              - Otherwise append as a new record.
          * Perform an atomic rewrite (temporary file + replace).
      - If stage2=True:
          * Add "stage2": True to each item being written

    Notes:
      - Accumulation:
          * "prompt_tokens" and "completion_tokens" are summed as integers.
          * "cost" is summed as float.
          * Missing or non-numeric values are treated as 0.
      - Other fields are taken from the new item (full replacement semantics except for accumulation fields).
      - Malformed lines in the existing JSONL are skipped with a warning.

    Args:
        path: Target JSONL filepath.
        data: List of dict records to write.
        append: Ignored in upsert mode (key != None). Kept for backward compatibility.
        key: Upsert key field name (e.g., "entry_point", "task_id"). If None, no upsert.
        accum_fields: Iterable of field names to accumulate when upserting.
        stage2: If True, adds "stage2": True to each item (indicates second pass processing).

    Returns:
        None
    """

    def _to_int(x):
        try:
            # allow numeric strings
            return int(x)
        except Exception:
            try:
                return int(float(x))
            except Exception:
                return 0

    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    def _accumulate_fields(old_rec: dict, new_rec: dict) -> dict:
        """Return a merged copy of new_rec where accum_fields are old+new."""
        merged = dict(new_rec)  # start from new (new fields overwrite)
        for f in accum_fields:
            if f == "cost":
                merged[f] = _to_float(old_rec.get(f, 0.0)) + _to_float(new_rec.get(f, 0.0))
            elif f in ("prompt_tokens", "completion_tokens"):
                merged[f] = _to_int(old_rec.get(f, 0)) + _to_int(new_rec.get(f, 0))
            else:
                # generic numeric add as float
                merged[f] = _to_float(old_rec.get(f, 0.0)) + _to_float(new_rec.get(f, 0.0))
        return merged

    # Add stage2 flag to all items if specified
    if stage2:
        data = [dict(item, stage2=True) for item in data]

    # Case 1: no upsert key -> original behavior
    if key is None:
        mode = 'a' if append else 'w'
        with jsonlines.open(path, mode=mode) as writer:
            for item in data:
                writer.write(item)
        return

    # Case 2: upsert by key with accumulation
    existing = []
    key_to_last_idx = {}

    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    print(f"[write_jsonl:warn] Skipping malformed JSON at line {lineno}", file=sys.stderr)
                    continue
                existing.append(rec)
                if isinstance(rec, dict) and key in rec:
                    key_to_last_idx[rec[key]] = len(existing) - 1

    # Upsert each incoming item
    for item in data:
        if not isinstance(item, dict):
            continue
        if key in item and item[key] is not None:
            kv = item[key]
            if kv in key_to_last_idx:
                idx = key_to_last_idx[kv]
                # accumulate selected fields; replace others with new item
                existing[idx] = _accumulate_fields(existing[idx], item)
            else:
                existing.append(item)
                key_to_last_idx[kv] = len(existing) - 1
        else:
            # No usable key -> append as-is
            existing.append(item)

    # Atomic rewrite
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False,
                                     dir=os.path.dirname(path) or ".",
                                     prefix=os.path.basename(path) + '.tmp.',
                                     encoding='utf-8') as tmp:
        tmp_path = tmp.name
        with jsonlines.Writer(tmp) as writer:
            for rec in existing:
                writer.write(rec)

    # Complete the atomic write by replacing the target file
    os.replace(tmp_path, path)



def read_jsonl_gz(path: str) -> List[dict]:
    if not path.endswith(".jsonl.gz"):
        raise ValueError(f"File `{path}` is not a jsonl.gz file.")
    with gzip.open(path, "rt") as f:
        data = [json.loads(line) for line in f]
    return data


# generator that returns the item and the index in the dataset.
# if the results_path exists, it will skip all items that have been processed
# before.
def enumerate_resume(dataset, results_path):
    if not os.path.exists(results_path):
        for i, item in enumerate(dataset):
            yield i, item
    else:
        count = 0
        with jsonlines.open(results_path) as reader:
            for item in reader:
                count += 1

        for i, item in enumerate(dataset):
            # skip items that have been processed before
            if i < count:
                continue
            yield i, item

            
def enumerate_resume_dotbank(dataset, results_path):

    if not os.path.exists(results_path):
        for i, item in enumerate(dataset):
            yield i, item
    else:

        with open(results_path) as f:
            results_data = [json.loads(line) for line in f]
        
        count = 0
        # Determine the primary key to use for matching
        if 'entry_point' in dataset[0]:
            primary_key = 'entry_point'
        elif 'task_id' in dataset[0]:
            primary_key = 'task_id'
        elif 'question_id' in dataset[0]:
            primary_key = 'question_id'
        else:
            primary_key = 'name'

        for i, item in enumerate(dataset):
            if item[primary_key] == results_data[-1][primary_key]:
                count = i
                break

        for i, item in enumerate(dataset):
            # skip items that have been processed before
            if i < count:
                continue
            yield i, item

def resume_success_count(dataset) -> int:
    count = 0
    for item in dataset:
        if "is_solved" in item and item["is_solved"]:
            count += 1
    return count




