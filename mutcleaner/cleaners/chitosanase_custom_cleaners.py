from __future__ import annotations

import io
from pathlib import Path
import pandas as pd
from typing import TYPE_CHECKING

from .basic_cleaners import (
    add_columns,
    extract_and_rename_columns,
    convert_data_types,
    filter_and_clean_data,
    subtract_labels_by_wt,
)
from ..core.pipeline import pipeline_step

if TYPE_CHECKING:
    from typing import List

__all__ = ["parse_chitosanase_raw_file"]


def __dir__() -> List[str]:
    return __all__


@pipeline_step
def parse_chitosanase_raw_file(file_path: str | Path, wt_separator: str = '">wt') -> pd.DataFrame:
    """Parse a raw Chitosanase input file and produce an intermediate DataFrame.

    The Chitosanase raw file format expected by this step is a small CSV
    block followed by a wild-type sequence. The two blocks are separated by
    a short marker token (default '" >wt'). This step extracts the CSV,
    normalizes the mutation column, retains the numerical label column
    required downstream, and performs WT subtraction to produce a DataFrame
    containing per-mutation delta-labels (`dTm`).

    Parameters
    ----------
    file_path : str | pathlib.Path
        Path to the raw Chitosanase input file. The file should contain a
        CSV section then the WT sequence separated by ``wt_separator``.
    wt_separator : str, optional
        Substring that separates CSV and WT sequence blocks (default '\">wt').

    Returns
    -------
    pd.DataFrame
        Intermediate DataFrame with at least the columns ``mut_info`` and
        ``dTm`` plus the added metadata columns ``name``, ``wt_seq`` and
        ``sequence``. The ``dTm`` column is a float representing the label
        value after WT subtraction.

    Raises
    ------
    ValueError
        If the expected WT separator is not found or if WT subtraction fails
        for one or more rows.

    Examples
    --------
    >>> from pathlib import Path
    >>> df = parse_chitosanase_raw_file(Path("/path/to/Chitosanase_Dataset.csv"))
    >>> list(df.columns)
    ['mut_info', 'dTm', 'name', 'wt_seq', 'sequence']
    """
    with open(file_path, "r") as f:
        raw_text = f.read()

    if wt_separator in raw_text:
        parts = raw_text.split(wt_separator)
        csv_text = parts[0].strip()
        wt_seq = parts[1].replace('"', "").replace(",", "").strip()
        wt_seq = "".join(wt_seq.split())
    else:
        raise ValueError(f"Cannot find WT sequence separator '{wt_separator}' in the expected format.")

    df = pd.read_csv(io.StringIO(csv_text))
    df["aa_mut"] = df["aa_mut"].astype(str).str.replace('"', "").str.strip()
    df = filter_and_clean_data(df, drop_na_columns=["Tm"])
    df = convert_data_types(df, {"Tm": "float"})

    df = extract_and_rename_columns(
        df,
        column_mapping={
            "aa_mut": "mut_info",
            "Tm": "Tm",
        },
    )
    df = add_columns(
        df,
        columns_to_add={
            "name": "Chitosanase",
            "wt_seq": wt_seq,
            "sequence": wt_seq,
        },
    )

    subtraction_result = subtract_labels_by_wt(
        dataset=df,
        name_column="name",
        label_columns="Tm",
        mutation_column="mut_info",
        wt_identifier="WT",
        in_place=True,
        drop_wt_row=True,
    )

    successful_df = subtraction_result.main
    failed_df = subtraction_result.side.get("failed", pd.DataFrame())

    if not failed_df.empty:
        raise ValueError("Failed to subtract WT Tm for Chitosanase rows: " f"{failed_df.get('error_message', pd.Series(dtype=str)).tolist()}")

    successful_df = successful_df.rename(columns={"Tm": "dTm"})
    successful_df = convert_data_types(successful_df, {"dTm": "float"})
    return successful_df
