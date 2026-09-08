#!/usr/bin/env python3

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
import pandas as pd


DNA_COMP = str.maketrans({
    "A": "T", "C": "G", "G": "C", "T": "A", "N": "N"
})


def revcomp(seq):
    return str(seq).upper().translate(DNA_COMP)[::-1]


def hamming(a, b):
    a = str(a).upper()
    b = str(b).upper()
    if len(a) != len(b):
        return None
    return sum(x != y for x, y in zip(a, b))


def orientation_invariant_hamming(a, b):
    """
    Distance relevant to the existing demux, which tests an expected index
    in both direct and reverse-complement orientation.

    Returns min(H(a,b), H(a,revcomp(b))).
    Different lengths cannot transform into each other by substitutions only,
    so return None.
    """
    d1 = hamming(a, b)
    if d1 is None:
        return None
    d2 = hamming(a, revcomp(b))
    return min(d1, d2)


def add_distances(a, b):
    if a is None or b is None:
        return None
    return a + b


def fmt(x):
    return "NA" if x is None else int(x)


def detect_delimiter(path):
    with open(path, "r", newline="") as fh:
        sample = fh.read(4096)
    return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter


def read_input(path, has_header=False):
    sep = detect_delimiter(path)

    if has_header:
        df = pd.read_csv(path, sep=sep, dtype=str)

        # Normalize column names but require the expected three fields.
        normalized = {str(c).strip().lower(): c for c in df.columns}
        required = ["sample", "f_seq", "r_seq"]
        missing = [c for c in required if c not in normalized]
        if missing:
            raise ValueError(
                "Header input must contain columns: sample, f_seq, r_seq. "
                f"Missing: {', '.join(missing)}"
            )

        df = df[
            [normalized["sample"], normalized["f_seq"], normalized["r_seq"]]
        ].copy()
        df.columns = ["sample", "f_seq", "r_seq"]

    else:
        df = pd.read_csv(path, sep=sep, header=None, dtype=str)

        if df.shape[1] < 3:
            raise ValueError(
                f"Input has {df.shape[1]} columns; 3 are required: "
                "sample,f_seq,r_seq"
            )

        df = df.iloc[:, :3].copy()
        df.columns = ["sample", "f_seq", "r_seq"]

    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    for c in ["f_seq", "r_seq"]:
        df[c] = df[c].str.upper()

    # Reject completely duplicated sample rows because they add no useful
    # information and would make downstream reporting confusing.
    df = df.drop_duplicates(
        subset=["sample", "f_seq", "r_seq"],
        keep="first"
    ).reset_index(drop=True)

    return df, sep

def pair_distance(row1, row2):
    """
    Minimum substitutions needed to transform one valid physical two-index pair
    into the other under the matching behavior of the current demux.

    Because the demux searches f_index and r_index at BOTH read ends, use the
    safer of two mappings:

      same-role:
          F1 -> F2  and  R1 -> R2

      crossed-role:
          F1 -> R2  and  R1 -> F2

    Each single-index distance is orientation invariant because the demux tests
    both direct and reverse-complement expected index sequences.
    """
    ff = orientation_invariant_hamming(row1["f_seq"], row2["f_seq"])
    rr = orientation_invariant_hamming(row1["r_seq"], row2["r_seq"])
    fr = orientation_invariant_hamming(row1["f_seq"], row2["r_seq"])
    rf = orientation_invariant_hamming(row1["r_seq"], row2["f_seq"])

    same = add_distances(ff, rr)
    crossed = add_distances(fr, rf)

    candidates = [x for x in (same, crossed) if x is not None]
    minimum = min(candidates) if candidates else None

    if minimum is None:
        mapping = "incomparable_lengths"
    elif same is not None and minimum == same:
        mapping = "same_role"
        if crossed is not None and crossed == same:
            mapping = "same_role;crossed_role"
    else:
        mapping = "crossed_role"

    return {
        "h_F1_F2": ff,
        "h_R1_R2": rr,
        "same_role_pair_distance": same,
        "h_F1_R2": fr,
        "h_R1_F2": rf,
        "crossed_role_pair_distance": crossed,
        "minimum_pair_distance": minimum,
        "minimum_mapping": mapping,
    }


def build_conflicts(df, max_pair_errors):
    rows = []
    conflicts = {i: set() for i in range(len(df))}

    for i in range(len(df)):
        a = df.iloc[i]
        for j in range(i + 1, len(df)):
            b = df.iloc[j]
            d = pair_distance(a, b)
            md = d["minimum_pair_distance"]

            # User's criterion:
            # if <= N substitutions can turn one valid pair exactly into another
            # valid pair, they must not be present in the same run.
            is_conflict = md is not None and md <= max_pair_errors

            if is_conflict:
                conflicts[i].add(j)
                conflicts[j].add(i)

            rows.append({
                "row1": i + 1,
                "sample1": a["sample"],
                "f_seq1": a["f_seq"],
                "r_seq1": a["r_seq"],
                "row2": j + 1,
                "sample2": b["sample"],
                "f_seq2": b["f_seq"],
                "r_seq2": b["r_seq"],
                "h_F1_F2": fmt(d["h_F1_F2"]),
                "h_R1_R2": fmt(d["h_R1_R2"]),
                "same_role_pair_distance": fmt(d["same_role_pair_distance"]),
                "h_F1_R2": fmt(d["h_F1_R2"]),
                "h_R1_F2": fmt(d["h_R1_F2"]),
                "crossed_role_pair_distance": fmt(d["crossed_role_pair_distance"]),
                "minimum_pair_distance": fmt(md),
                "minimum_mapping": d["minimum_mapping"],
                "max_pair_errors": max_pair_errors,
                "can_transform_within_error_budget": "YES" if is_conflict else "NO",
                "safe_together": "NO" if is_conflict else "YES",
            })

    return pd.DataFrame(rows), conflicts


def bron_kerbosch_maximal_independent_sets(conflicts, max_sets=10000):
    """
    Enumerate maximal independent sets of the conflict graph by finding maximal
    cliques in its complement.

    A returned set is MAXIMAL: no additional pair can be added without creating
    a conflict. The sets are later sorted largest-first.
    """
    vertices = set(conflicts)
    compat = {
        v: vertices - {v} - conflicts[v]
        for v in vertices
    }

    results = []

    def bk(R, P, X):
        if len(results) >= max_sets:
            return
        if not P and not X:
            results.append(tuple(sorted(R)))
            return

        union = P | X
        if union:
            u = max(union, key=lambda z: len(P & compat[z]))
            candidates = list(P - compat[u])
        else:
            candidates = list(P)

        for v in candidates:
            bk(R | {v}, P & compat[v], X & compat[v])
            P.remove(v)
            X.add(v)
            if len(results) >= max_sets:
                return

    bk(set(), set(vertices), set())
    results = sorted(set(results), key=lambda s: (-len(s), s))
    return results


def greedy_disjoint_run_partition(conflicts):
    """
    Simple practical partition: assign every input pair to one safe run.

    This is a greedy graph-coloring heuristic, not guaranteed to use the absolute
    minimum number of runs, but every produced run is conflict-free.
    """
    vertices = sorted(conflicts, key=lambda v: (-len(conflicts[v]), v))
    runs = []

    for v in vertices:
        placed = False
        for run in runs:
            if all(u not in conflicts[v] for u in run):
                run.append(v)
                placed = True
                break
        if not placed:
            runs.append([v])

    return [tuple(sorted(run)) for run in runs]


def subsets_to_table(subsets, df, prefix="subset"):
    rows = []
    for k, subset in enumerate(subsets, start=1):
        for pos, idx in enumerate(subset, start=1):
            r = df.iloc[idx]
            rows.append({
                "subset_id": f"{prefix}_{k}",
                "subset_size": len(subset),
                "member_no": pos,
                "input_row": idx + 1,
                "sample": r["sample"],
                "f_seq": r["f_seq"],
                "r_seq": r["r_seq"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipetting convenience scoring
# ---------------------------------------------------------------------------

PLATE_WELL_RE = re.compile(
    r"(?P<plate>[A-Za-z0-9]+)_(?P<row>[A-Ha-h])(?P<col>0?[1-9]|1[0-2])(?:$|_)"
)
ROW_ORDER = {row: i for i, row in enumerate("ABCDEFGH")}


def extract_plate_wells(value):
    """
    Extract physical 96-well positions encoded like:
        A_A01
        A_B01
        C_G02
        C_G02_UDP0207

    Returns (plate, row, column) tuples.
    """
    value = str(value)
    hits = []
    for m in PLATE_WELL_RE.finditer(value):
        hits.append(
            (
                m.group("plate"),
                m.group("row").upper(),
                int(m.group("col")),
            )
        )
    return hits


def physical_locations_for_index_pair(row):
    """
    Extract the physical 96-well position from the sample name only.

    Examples:
        A_A01
        A_B01
        C_G02_UDP0207

    become:
        plate A, well A01
        plate A, well B01
        plate C, well G02
    """
    locations = set(extract_plate_wells(row["sample"]))
    return sorted(locations, key=lambda x: (x[0], x[2], ROW_ORDER[x[1]]))


def pipette_score(subset, df):
    """
    Rank equally large SAFE subsets by 8-channel pipetting convenience.

    Priority:
      1. more complete A-H columns
      2. more wells in complete columns
      3. more vertically adjacent wells
      4. longer contiguous vertical runs
      5. fewer holes inside occupied columns
      6. fewer partial columns
      7. fewer total columns touched
      8. fewer plates touched
    """
    locations = set()
    for idx in subset:
        locations.update(physical_locations_for_index_pair(df.iloc[idx]))

    by_col = defaultdict(set)
    for plate, row, col in locations:
        by_col[(plate, col)].add(row)

    complete_columns = 0
    wells_in_complete_columns = 0
    adjacent_pairs = 0
    contiguous_run_score = 0
    holes = 0
    partial_columns = 0

    for rows in by_col.values():
        row_nums = sorted(ROW_ORDER[r] for r in rows)
        n = len(row_nums)

        if n == 8:
            complete_columns += 1
            wells_in_complete_columns += 8
        elif n > 0:
            partial_columns += 1

        adjacent_pairs += sum(
            1 for a, b in zip(row_nums, row_nums[1:]) if b == a + 1
        )

        if row_nums:
            runs = []
            current = 1
            for a, b in zip(row_nums, row_nums[1:]):
                if b == a + 1:
                    current += 1
                else:
                    runs.append(current)
                    current = 1
            runs.append(current)
            contiguous_run_score += sum(r * r for r in runs)
            holes += (row_nums[-1] - row_nums[0] + 1) - n

    plates = {plate for plate, _, _ in locations}

    return (
        complete_columns,
        wells_in_complete_columns,
        adjacent_pairs,
        contiguous_run_score,
        -holes,
        -partial_columns,
        -len(by_col),
        -len(plates),
    )


def pipette_score_details(subset, df):
    locations = set()
    for idx in subset:
        locations.update(physical_locations_for_index_pair(df.iloc[idx]))

    by_col = defaultdict(set)
    for plate, row, col in locations:
        by_col[(plate, col)].add(row)

    complete_columns = 0
    adjacent_pairs = 0
    holes = 0
    partial_columns = 0

    for rows in by_col.values():
        row_nums = sorted(ROW_ORDER[r] for r in rows)
        if len(row_nums) == 8:
            complete_columns += 1
        elif row_nums:
            partial_columns += 1

        adjacent_pairs += sum(
            1 for a, b in zip(row_nums, row_nums[1:]) if b == a + 1
        )
        if row_nums:
            holes += (row_nums[-1] - row_nums[0] + 1) - len(row_nums)

    return {
        "complete_8channel_columns": complete_columns,
        "vertical_adjacent_pairs": adjacent_pairs,
        "partial_columns": partial_columns,
        "holes_within_columns": holes,
        "columns_touched": len(by_col),
        "plates_touched": len({plate for plate, _, _ in locations}),
        "recognized_plate_wells": len(locations),
    }


def choose_easiest_to_pipette(largest_subsets, df):
    """
    Choose one deterministic easiest-to-pipette solution among the largest
    safe subsets.
    """
    if not largest_subsets:
        return None, pd.DataFrame()

    ranked = []
    for n, subset in enumerate(largest_subsets, start=1):
        ranked.append({
            "original_largest_subset_no": n,
            "subset": subset,
            "score": pipette_score(subset, df),
            **pipette_score_details(subset, df),
        })

    ranked.sort(
        key=lambda x: (x["score"], tuple(-v for v in x["subset"])),
        reverse=True,
    )

    score_rows = []
    for rank, item in enumerate(ranked, start=1):
        score_rows.append({
            "pipette_rank": rank,
            "largest_subset_id":
                f"largest_safe_subset_{item['original_largest_subset_no']}",
            "subset_size": len(item["subset"]),
            "complete_8channel_columns": item["complete_8channel_columns"],
            "vertical_adjacent_pairs": item["vertical_adjacent_pairs"],
            "partial_columns": item["partial_columns"],
            "holes_within_columns": item["holes_within_columns"],
            "columns_touched": item["columns_touched"],
            "plates_touched": item["plates_touched"],
            "recognized_plate_wells": item["recognized_plate_wells"],
        })

    return ranked[0], pd.DataFrame(score_rows)


def easiest_subset_to_table(best, df):
    """
    Lab-facing table sorted plate -> column -> row so that
    A01..H01, then A02..H02, etc. are consecutive.
    """
    if best is None:
        return pd.DataFrame()

    rows = []
    for idx in best["subset"]:
        r = df.iloc[idx]
        locs = physical_locations_for_index_pair(r)

        if locs:
            plate, well_row, well_col = locs[0]
            well = f"{well_row}{well_col:02d}"
            sort_key = (plate, well_col, ROW_ORDER[well_row], idx)
        else:
            plate = ""
            well = ""
            sort_key = ("~", 999, 999, idx)

        rows.append({
            "_sort_key": sort_key,
            "pipette_plate": plate,
            "pipette_well": well,
            "sample": r["sample"],
            "f_seq": r["f_seq"],
            "r_seq": r["r_seq"],
        })

    rows.sort(key=lambda x: x["_sort_key"])

    for n, row in enumerate(rows, start=1):
        row["pipette_order"] = n
        del row["_sort_key"]

    cols = [
        "pipette_order",
        "pipette_plate",
        "pipette_well",
        "sample",
        "f_seq",
        "r_seq",
    ]
    return pd.DataFrame(rows)[cols]

def plate_name_for_row(row):
    """
    Return the plate parsed from sample name, e.g.
        C_G02_UDP0207 -> C

    Return None when no <plate>_<A01-H12> pattern is recognized.
    """
    locs = physical_locations_for_index_pair(row)
    if not locs:
        return None
    return locs[0][0]


def analyze_one_plate(df_plate, plate_name, outdir, max_pair_errors, max_subsets):
    """
    Run the full conflict/subset/pipetting analysis independently for one plate.
    """
    plate_dir = outdir / f"plate_{plate_name}"
    plate_dir.mkdir(parents=True, exist_ok=True)

    # Work with local row numbering inside the plate.
    df_plate = df_plate.reset_index(drop=True)

    pairwise, conflicts = build_conflicts(df_plate, max_pair_errors)

    conflict_df = pairwise[
        pairwise["can_transform_within_error_budget"] == "YES"
    ].copy()
    safe_pairwise_df = pairwise[
        pairwise["can_transform_within_error_budget"] == "NO"
    ].copy()

    maximal_subsets = bron_kerbosch_maximal_independent_sets(
        conflicts, max_sets=max_subsets
    )
    maximal_table = subsets_to_table(
        maximal_subsets, df_plate, "safe_subset"
    )

    largest_size = max((len(s) for s in maximal_subsets), default=0)
    largest_subsets = [
        s for s in maximal_subsets if len(s) == largest_size
    ]
    largest_table = subsets_to_table(
        largest_subsets, df_plate, "largest_safe_subset"
    )

    best_pipette, pipette_scores = choose_easiest_to_pipette(
        largest_subsets, df_plate
    )
    easiest_pipette_table = easiest_subset_to_table(
        best_pipette, df_plate
    )

    run_partition = greedy_disjoint_run_partition(conflicts)
    partition_table = subsets_to_table(
        run_partition, df_plate, "run"
    )

    node_rows = []
    for i in range(len(df_plate)):
        r = df_plate.iloc[i]
        node_rows.append({
            "input_row_within_plate": i + 1,
            "sample": r["sample"],
            "f_seq": r["f_seq"],
            "r_seq": r["r_seq"],
            "n_conflicting_pairs": len(conflicts[i]),
            "can_share_one_run_with_all_others": (
                "YES" if len(conflicts[i]) == 0 else "NO"
            ),
        })
    node_df = pd.DataFrame(node_rows)

    # Write per-plate outputs.
    df_plate.to_csv(
        plate_dir / "plate_input_indices.tsv",
        sep="\t", index=False
    )
    pairwise.to_csv(
        plate_dir / "all_pairwise_pair_distances.tsv",
        sep="\t", index=False
    )
    conflict_df.to_csv(
        plate_dir / "conflicting_index_pairs.tsv",
        sep="\t", index=False
    )
    safe_pairwise_df.to_csv(
        plate_dir / "safe_pairwise_combinations.tsv",
        sep="\t", index=False
    )
    node_df.to_csv(
        plate_dir / "per_pair_conflict_counts.tsv",
        sep="\t", index=False
    )
    maximal_table.to_csv(
        plate_dir / "maximal_safe_subsets.tsv",
        sep="\t", index=False
    )
    largest_table.to_csv(
        plate_dir / "largest_safe_subsets.tsv",
        sep="\t", index=False
    )
    pipette_scores.to_csv(
        plate_dir / "largest_safe_subsets_pipetting_scores.tsv",
        sep="\t", index=False
    )
    easiest_pipette_table.to_csv(
        plate_dir / "easiest_to_pipette_largest_safe_subset.tsv",
        sep="\t", index=False
    )
    partition_table.to_csv(
        plate_dir / "suggested_run_partition.tsv",
        sep="\t", index=False
    )

    best_subset_name = ""
    complete_cols = 0
    partial_cols = 0
    recognized_wells = 0

    if best_pipette is not None:
        best_subset_name = (
            f"largest_safe_subset_"
            f"{best_pipette['original_largest_subset_no']}"
        )
        details = pipette_score_details(
            best_pipette["subset"], df_plate
        )
        complete_cols = details["complete_8channel_columns"]
        partial_cols = details["partial_columns"]
        recognized_wells = details["recognized_plate_wells"]

    return {
        "plate": plate_name,
        "n_index_pairs": len(df_plate),
        "n_conflicting_pair_comparisons": len(conflict_df),
        "n_maximal_safe_subsets": len(maximal_subsets),
        "largest_safe_subset_size": largest_size,
        "n_largest_safe_subsets": len(largest_subsets),
        "chosen_pipette_subset": best_subset_name,
        "complete_8channel_columns": complete_cols,
        "partial_columns": partial_cols,
        "recognized_plate_wells_in_chosen_subset": recognized_wells,
        "greedy_run_partition_count": len(run_partition),
        "subset_enumeration_hit_limit": (
            "YES" if len(maximal_subsets) >= max_subsets else "NO"
        ),
    }


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Split a 3-column ONT index-pair list by physical 96-well plate, "
            "then independently find safe index-pair subsets within each plate. "
            "Safety uses pair-level Hamming distance and the existing demux's "
            "direct/reverse-complement matching behavior."
        )
    )
    ap.add_argument(
        "-i", "--input", required=True,
        help="CSV/TSV with: sample,f_seq,r_seq"
    )
    ap.add_argument(
        "--header", action="store_true",
        help="Input has a header row"
    )
    ap.add_argument(
        "--max-pair-errors", type=int, default=2,
        help=(
            "Maximum total substitutions across the TWO indices that must not "
            "be able to convert one valid pair into another [default: 2]"
        ),
    )
    ap.add_argument(
        "-o", "--outdir", default="index_pair_check_per_plate",
        help="Output directory"
    )
    ap.add_argument(
        "--max-subsets", type=int, default=10000,
        help=(
            "Maximum maximal safe subsets to enumerate PER PLATE "
            "[default: 10000]"
        )
    )

    args = ap.parse_args()

    if args.max_pair_errors < 0:
        raise ValueError("--max-pair-errors must be >= 0")
    if args.max_subsets < 1:
        raise ValueError("--max-subsets must be >= 1")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df, sep = read_input(args.input, args.header)

    # Parse plate from sample name and preserve the original input row.
    df = df.copy()
    df["original_input_row"] = range(1, len(df) + 1)
    df["plate"] = [
        plate_name_for_row(df.iloc[i])
        for i in range(len(df))
    ]

    recognized_df = df[df["plate"].notna()].copy()
    unrecognized_df = df[df["plate"].isna()].copy()

    if recognized_df.empty:
        raise ValueError(
            "No sample names contained a recognized <plate>_<A01-H12> "
            "pattern. Example recognized names: A_A01, C_G02_UDP0207."
        )

    # Record how the input was split.
    split_table = recognized_df[
        ["original_input_row", "plate", "sample", "f_seq", "r_seq"]
    ].copy()
    split_table.to_csv(
        outdir / "indices_split_by_plate.tsv",
        sep="\t", index=False
    )

    if not unrecognized_df.empty:
        unrecognized_df[
            ["original_input_row", "sample", "f_seq", "r_seq"]
        ].to_csv(
            outdir / "unrecognized_plate_samples.tsv",
            sep="\t", index=False
        )

    summaries = []
    lab_tables = []

    # Analyze each plate independently.
    for plate_name in sorted(recognized_df["plate"].unique()):
        df_plate = recognized_df[
            recognized_df["plate"] == plate_name
        ][["sample", "f_seq", "r_seq"]].copy()

        summary = analyze_one_plate(
            df_plate=df_plate,
            plate_name=plate_name,
            outdir=outdir,
            max_pair_errors=args.max_pair_errors,
            max_subsets=args.max_subsets,
        )
        summaries.append(summary)

        lab_file = (
            outdir
            / f"plate_{plate_name}"
            / "easiest_to_pipette_largest_safe_subset.tsv"
        )
        if lab_file.exists():
            lab_df = pd.read_csv(lab_file, sep="\t", dtype=str)
            if not lab_df.empty:
                lab_df.insert(0, "plate_group", plate_name)
                lab_tables.append(lab_df)

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(
        outdir / "per_plate_summary.tsv",
        sep="\t", index=False
    )

    # One convenient combined lab file containing the selected best subset
    # from every plate, but without mixing plates during the optimization.
    if lab_tables:
        combined_lab = pd.concat(lab_tables, ignore_index=True)
        combined_lab.to_csv(
            outdir / "LAB_easiest_safe_subset_per_plate.tsv",
            sep="\t", index=False
        )
    else:
        combined_lab = pd.DataFrame()

    print("Done.")
    print(f"Detected delimiter: {repr(sep)}")
    print(f"Total input index pairs: {len(df)}")
    print(f"Recognized plate samples: {len(recognized_df)}")
    print(f"Unrecognized plate samples: {len(unrecognized_df)}")
    print(f"Number of plates: {recognized_df['plate'].nunique()}")
    print(
        f"Error budget: <= {args.max_pair_errors} total substitutions "
        "across both indices"
    )
    print()

    for s in summaries:
        print(
            f"Plate {s['plate']}: "
            f"{s['n_index_pairs']} pairs -> "
            f"largest safe subset {s['largest_safe_subset_size']} pairs; "
            f"{s['complete_8channel_columns']} complete 8-channel columns"
        )

    print()
    print("Main outputs:")
    print(f"  {outdir / 'indices_split_by_plate.tsv'}")
    print(f"  {outdir / 'per_plate_summary.tsv'}")
    print(f"  {outdir / 'LAB_easiest_safe_subset_per_plate.tsv'}")
    print()
    print("Each plate also has its own directory:")
    for plate_name in sorted(recognized_df["plate"].unique()):
        print(f"  {outdir / f'plate_{plate_name}'}")




if __name__ == "__main__":
    main()
