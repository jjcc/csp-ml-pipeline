
#!/usr/bin/env python3
"""
a05_merge_datasets.py — Step 5 (optional): Merge multiple enriched datasets incrementally for walk-forward training.
"""
import os
import sys
from pathlib import Path
import pandas as pd
from datetime import time
from service.env_config import get_derived_file, getenv, config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():

    # inputs
    input_dir = getenv("COMMON_OUTPUT_DIR", "output")
    input_dir = os.path.join(input_dir, "data_prep")
    out_dir = getenv("COMMON_OUTPUT_DIR", "output")
    out_dir = os.path.join(out_dir, "data_merged")
    os.makedirs(out_dir, exist_ok=True)


    common_configs = config.get_common_configs_raw()
    # Get all the group tags
    info_by_tag = {} # dictionary ->  tag: (basic_csv, cutoff_date)
    group_tags  = [] # list of tags in order -> [orig, a, b, c ...]
    for k, v in common_configs.items():
        #print(f"  {k}: {v}")
        basic_csv = v.get("data_basic_csv", "N/A")
        file_name = basic_csv.replace(".csv", "")
        file_name_seg = file_name.split("_")
        group_tag = file_name_seg[file_name_seg.index("raw") +1]
        print(f"Processing config group: {k}, tag: {group_tag}")

        cutoff_date = v.get("cutoff_date", None)
        info_by_tag[group_tag] = (basic_csv, cutoff_date)
        group_tags.append(group_tag)

    # get the combination of sequential tags -> orig, orig+a, orig+a+b, ...
    for i in range(len(group_tags)):
        tags = group_tags[:i+1]
        print(f"Merging tags: {tags}")
        basic_csvs = [info_by_tag[t][0] for t in tags]
        print(f"  Basic CSVs: {basic_csvs}")
        last_tag = tags[-1]
        # get the cutoff date from the last tag's config
        last_config = info_by_tag[last_tag]
        cutoff_date = last_config[1]
        print(f"  Cutoff date of {last_tag}: {cutoff_date}")

        dfs = []
        for bc in basic_csvs:
            fpath = get_derived_file(bc)[0]
            fpath = os.path.join(input_dir, os.path.basename(fpath))
            print(f"    Loading {fpath}")
            dfi = pd.read_csv(fpath)
            dfs.append(dfi)
        d = pd.concat(dfs, ignore_index=True)
        print(f"  Merged dataset rows before filtering: {len(d)}")
        # filter out daysToExpiration <= 14
        d = d[d["daysToExpiration"] <= 14]
        print(f"  Merged dataset rows after daysToExpiration <= 14 filter: {len(d)}")

        # filter by cutoff date if specified
        if cutoff_date is not None:
            before_count = len(d)
            d = d[pd.to_datetime(d["tradeTime"], errors="coerce") <= pd.to_datetime(cutoff_date)]
            after_count = len(d)
            print(f"  Applied cutoff date {cutoff_date}, filtered out {before_count - after_count} rows, remaining {after_count} rows")
        out_csv = os.path.join(out_dir, f"merged_with_gex_macro_{''.join(tags)}.csv")
        d.to_csv(out_csv, index=False)
        #print(f"  Merged dataset rows: {len(d)}")






def merge_dataset_with_feat():
    pass



if __name__ == "__main__":
    main()
