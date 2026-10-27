"""Loading and validating uploaded review files (CSV or Excel)."""

from __future__ import annotations

import io

import pandas as pd

ID_ALIASES = ("id", "review_id", "comment_id", "reviewid", "commentid")
COMMENT_ALIASES = ("comment", "comments", "review", "reviews", "text", "review_text", "feedback")


class DataError(ValueError):
    """Raised when an uploaded file cannot be turned into an id/comment table."""


def _read_raw(file_name: str, data: bytes) -> pd.DataFrame:
    name = file_name.lower()
    buffer = io.BytesIO(data)
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(buffer)
    if name.endswith((".csv", ".txt", ".tsv")):
        sep = "\t" if name.endswith(".tsv") else None
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                buffer.seek(0)
                return pd.read_csv(buffer, sep=sep, engine="python", encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise DataError("Could not decode the CSV file. Please save it as UTF-8.")
    raise DataError("Unsupported file type. Upload a .csv, .xlsx or .xls file.")


def _find_column(columns: list[str], aliases: tuple[str, ...]) -> str | None:
    lookup = {c.strip().lower().replace(" ", "_"): c for c in columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


def load_reviews(file_name: str, data: bytes) -> pd.DataFrame:
    """Return a clean DataFrame with exactly two columns: ``id`` and ``comment``.

    Column names are matched case-insensitively (``id``/``review_id`` and
    ``comment``/``review``/``text`` ...). If no names match but the file has two
    columns, the first is taken as id and the second as comment.
    """
    raw = _read_raw(file_name, data)
    if raw.empty:
        raise DataError("The uploaded file is empty.")

    columns = [str(c) for c in raw.columns]
    raw.columns = columns
    id_col = _find_column(columns, ID_ALIASES)
    comment_col = _find_column(columns, COMMENT_ALIASES)

    if comment_col is None and len(columns) == 2:
        id_col, comment_col = columns
    if comment_col is None:
        raise DataError(
            "Could not find a comment column. Name your columns 'id' and 'comment'."
        )

    df = pd.DataFrame({"comment": raw[comment_col]})
    df["id"] = raw[id_col] if id_col is not None else range(1, len(raw) + 1)
    df = df[["id", "comment"]]

    df["comment"] = df["comment"].astype("string").str.strip()
    df = df[df["comment"].notna() & (df["comment"] != "")]
    df["comment"] = df["comment"].astype(str)
    df["id"] = df["id"].astype(str)
    df = df.drop_duplicates(subset=["id", "comment"]).reset_index(drop=True)
    # Results are joined on id, so repeated ids get a suffix to stay unique.
    dup = df.groupby("id").cumcount()
    df.loc[dup > 0, "id"] = df.loc[dup > 0, "id"] + "_" + dup[dup > 0].astype(str)
    if df.empty:
        raise DataError("No non-empty comments were found in the file.")
    return df
