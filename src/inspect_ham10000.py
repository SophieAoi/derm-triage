"""Summarise the HAM10000 metadata: classes, images, lesions, patients, Fitzpatrick labels.

Reads the metadata file and lists image file names. Does not open images or train anything.

Usage:
    python src/inspect_ham10000.py
    python src/inspect_ham10000.py --data-dir data/ham10000 --metadata path/to/metadata.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ham10000"
REQUIRED_COLUMNS = ("lesion_id", "image_id", "dx")
METADATA_SUFFIXES = {".csv", ".tab", ".tsv"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PATIENT_KEYWORDS = ("patient",)
FITZPATRICK_KEYWORDS = ("fitzpatrick", "fst", "skin_type", "skin_tone")


def visible_files(data_dir):
    """Yield files under data_dir, skipping hidden files such as macOS ._ entries."""
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            yield path


def find_metadata(data_dir):
    matches = [
        path
        for path in visible_files(data_dir)
        if "metadata" in path.name.lower() and path.suffix.lower() in METADATA_SUFFIXES
    ]
    if not matches:
        sys.exit(f"No metadata file found under {data_dir}. Expected a name like HAM10000_metadata.csv.")
    if len(matches) > 1:
        listed = "\n  ".join(str(path) for path in matches)
        sys.exit(f"Found several metadata files, choose one with --metadata:\n  {listed}")
    return matches[0]


def load_metadata(path):
    sep = "," if path.suffix.lower() == ".csv" else "\t"
    # Read as strings so IDs and labels are reported exactly as written.
    df = pd.read_csv(path, sep=sep, dtype=str)
    df.columns = df.columns.str.strip()
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        sys.exit(f"{path} is missing {missing}. Columns found: {list(df.columns)}")
    return df


def matching_columns(df, keywords):
    """Return columns whose normalised name contains any keyword."""
    normalised = {column: column.lower().replace(" ", "_").replace("-", "_") for column in df.columns}
    return [column for column, name in normalised.items() if any(word in name for word in keywords)]


def print_section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def report_images(df, data_dir):
    print_section("Image counts")
    ids = set(df["image_id"].dropna())
    print(f"Metadata rows:            {len(df)}")
    print(f"Unique image_id values:   {len(ids)}")
    print(f"Repeated image_id rows:   {df['image_id'].dropna().duplicated().sum()}")

    files = [path for path in visible_files(data_dir) if path.suffix.lower() in IMAGE_SUFFIXES]
    stems = {path.stem for path in files}
    print(f"Image files on disk:      {len(files)}")
    print(f"Unique image file names:  {len(stems)}")
    print(f"In metadata, no file:     {len(ids - stems)}")
    print(f"File, not in metadata:    {len(stems - ids)}")


def report_classes(df):
    print_section("Class counts (dx)")
    table = pd.DataFrame(
        {
            "images": df["dx"].value_counts(dropna=False),
            "lesions": df.groupby("dx", dropna=False)["lesion_id"].nunique(),
        }
    ).sort_values("images", ascending=False)
    table["% of images"] = (100 * table["images"] / len(df)).round(1)
    print(table.to_string())


def report_lesions(df):
    print_section("Lesions")
    lesion_rows = df.dropna(subset=["lesion_id"])
    print(f"Unique lesion_id values:  {lesion_rows['lesion_id'].nunique()}")
    print(f"Rows without lesion_id:   {len(df) - len(lesion_rows)}")

    images_per_lesion = lesion_rows.groupby("lesion_id")["image_id"].nunique()
    print("\nLesions by number of images:")
    print(images_per_lesion.value_counts().sort_index().rename_axis("images").to_string(header=False))

    labels_per_lesion = lesion_rows.groupby("lesion_id")["dx"].nunique()
    print(f"\nLesions with more than one dx: {(labels_per_lesion > 1).sum()}")


def report_patients(df):
    print_section("Patients per lesion")
    columns = matching_columns(df, PATIENT_KEYWORDS)
    if not columns:
        print("Not available: the metadata has no patient ID column.")
        print("Images per lesion above is the closest measure. Lesions from the same patient cannot be linked.")
        return

    for column in columns:
        rows = df.dropna(subset=["lesion_id", column])
        print(f"Column: {column}")
        print(f"Unique patients:          {rows[column].nunique()}")
        print(f"Rows without a value:     {df[column].isna().sum()}")
        print("Lesions by number of patients:")
        print(rows.groupby("lesion_id")[column].nunique().value_counts().sort_index().to_string(header=False))
        print("Patients by number of lesions:")
        print(rows.groupby(column)["lesion_id"].nunique().value_counts().sort_index().to_string(header=False))


def report_fitzpatrick(df):
    print_section("Fitzpatrick labels")
    columns = matching_columns(df, FITZPATRICK_KEYWORDS)
    if not columns:
        print("None: no column name suggests a Fitzpatrick or skin-type label.")
        print(f"Columns checked: {list(df.columns)}")
        return

    for column in columns:
        print(f"Column: {column}")
        print(df[column].value_counts(dropna=False).sort_index().to_string(header=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--metadata", type=Path, help="metadata file, if it cannot be found automatically")
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        sys.exit(f"Data folder not found: {args.data_dir}")

    metadata_path = args.metadata or find_metadata(args.data_dir)
    df = load_metadata(metadata_path)

    print(f"Metadata file: {metadata_path}")
    print(f"Columns: {list(df.columns)}")
    report_images(df, args.data_dir)
    report_classes(df)
    report_lesions(df)
    report_patients(df)
    report_fitzpatrick(df)


if __name__ == "__main__":
    main()
