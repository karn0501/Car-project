"""
Cross-Source Entity Resolution & Fuzzy Deduplication Module (Phase 6).
Uses RapidFuzz token matching to identify and merge duplicate used car listings
scraped across multiple platforms (e.g. CarDekho, Spinny, Cars24).
"""

import re
import pandas as pd
try:
    from rapidfuzz import fuzz
except ImportError:
    from difflib import SequenceMatcher
    class fuzz:
        @staticmethod
        def token_set_ratio(s1, s2):
            return SequenceMatcher(None, str(s1).lower(), str(s2).lower()).ratio() * 100


class CrossSourceDeduplicator:
    """
    Fuzzy Deduplication Engine for cross-platform vehicle listing resolution.
    """

    def __init__(self, similarity_threshold: float = 85.0):
        self.similarity_threshold = similarity_threshold

    def normalize_title(self, text: str) -> str:
        if not text or pd.isnull(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def find_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        df_out["is_duplicate"] = False
        df_out["duplicate_group_id"] = None

        if len(df_out) == 0:
            return df_out

        group_cols = [c for c in ["manufacture_year", "city"] if c in df_out.columns]
        grouped = df_out.groupby(group_cols) if group_cols else [("all", df_out)]

        group_counter = 1

        for _, group_df in grouped:
            indices = group_df.index.tolist()
            n = len(indices)
            if n <= 1:
                continue

            for i in range(min(n, 50)):
                idx1 = indices[i]
                if df_out.loc[idx1, "is_duplicate"]:
                    continue

                t1 = self.normalize_title(
                    f"{df_out.loc[idx1].get('company_name', '')} {df_out.loc[idx1].get('model_name', '')} {df_out.loc[idx1].get('variant_name', '')}"
                )
                km1 = float(df_out.loc[idx1].get('km_driven', 0))

                for j in range(i + 1, min(n, 50)):
                    idx2 = indices[j]
                    if df_out.loc[idx2, "is_duplicate"]:
                        continue

                    t2 = self.normalize_title(
                        f"{df_out.loc[idx2].get('company_name', '')} {df_out.loc[idx2].get('model_name', '')} {df_out.loc[idx2].get('variant_name', '')}"
                    )
                    km2 = float(df_out.loc[idx2].get('km_driven', 0))

                    title_sim = fuzz.token_set_ratio(t1, t2)
                    km_diff = abs(km1 - km2)

                    if title_sim >= self.similarity_threshold and km_diff <= 1500:
                        df_out.loc[idx2, "is_duplicate"] = True
                        gid = f"DUP_{group_counter:05d}"
                        df_out.loc[idx1, "duplicate_group_id"] = gid
                        df_out.loc[idx2, "duplicate_group_id"] = gid
                        group_counter += 1

        return df_out

    def deduplicate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_dups = self.find_duplicates(df)
        df_clean = df_dups[~df_dups["is_duplicate"]].copy()
        return df_clean.drop(columns=["is_duplicate", "duplicate_group_id"], errors="ignore")
