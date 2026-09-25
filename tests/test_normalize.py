"""Unit tests for Phase 2 data normalization module."""

import unittest
import numpy as np
import pandas as pd

from src.normalize import (
    normalize_address,
    normalize_business_name,
    normalize_country,
    normalize_dataframe,
    normalize_text,
)


class TestNormalize(unittest.TestCase):
    """Test suite for Phase 2 normalization."""

    def test_normalize_text_cases(self):
        """Test text normalization on standard cases from requirements."""
        self.assertEqual(normalize_text("O'Reilly's Barbershop"), "o reilly s barbershop")
        self.assertEqual(normalize_text("  PRIME   MONEY  "), "prime money")
        self.assertEqual(normalize_text("B+ Retail Inc"), "b retail inc")
        self.assertEqual(normalize_text("राम मार्केटिंग प्राइवेट लिमिटेड"), "राम मार्केटिंग प्राइवेट लिमिटेड")
        self.assertEqual(normalize_text("Fractales Amis Groupe S.A.S"), "fractales amis groupe s a s")
        self.assertEqual(normalize_text("17560 Ellis Road, Tahlequah, OK"), "17560 ellis road tahlequah ok")
        self.assertEqual(normalize_text("63 R. DE DIEPPE, LILLE"), "63 r de dieppe lille")
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(None), "")

    def test_normalize_address_preserves_numbers(self):
        """Verify that numbers and complex address patterns are preserved."""
        raw_addr = "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi"
        normalized = normalize_address(raw_addr)
        self.assertIn("570", normalized)
        self.assertIn("13", normalized)
        self.assertIn("new delhi", normalized)
        self.assertIn("kh no", normalized)

    def test_normalize_business_name_preserves_legal_suffixes(self):
        """Verify that legal suffixes (LLC, Inc, Ltd, SARL, SAS) are not removed."""
        self.assertIn("llc", normalize_business_name("LLC Moncada Learning Center"))
        self.assertIn("sarl", normalize_business_name("Marina Ecole France Sarl"))
        self.assertIn("inc", normalize_business_name("Zephay Labs Inc"))
        self.assertIn("s a s", normalize_business_name("Fractales Amis Groupe S.A.S"))
        self.assertIn("sas", normalize_business_name("Fractales Amis Groupe SAS"))

    def test_normalize_country(self):
        """Verify country normalization handles US, India, France, and generic strings."""
        self.assertEqual(normalize_country("US"), "us")
        self.assertEqual(normalize_country("  India  "), "india")
        self.assertEqual(normalize_country("France"), "france")
        self.assertEqual(normalize_country("Germany"), "germany")
        self.assertEqual(normalize_country(""), "")
        self.assertEqual(normalize_country(None), "")

    def test_normalize_series(self):
        """Verify vectorized Series normalization matching scalar behavior."""
        raw_series = pd.Series([
            "O'Reilly's Barbershop",
            "  PRIME   MONEY  ",
            "B+ Retail Inc",
            "राम मार्केटिंग प्राइवेट लिमिटेड",
            "Fractales Amis Groupe S.A.S",
            "17560 Ellis Road, Tahlequah, OK",
            "63 R. DE DIEPPE, LILLE",
            "",
            None,
            np.nan,
        ])
        res = normalize_text(raw_series)
        self.assertEqual(len(res), 10)
        self.assertEqual(res.iloc[0], "o reilly s barbershop")
        self.assertEqual(res.iloc[1], "prime money")
        self.assertEqual(res.iloc[2], "b retail inc")
        self.assertEqual(res.iloc[3], "राम मार्केटिंग प्राइवेट लिमिटेड")
        self.assertEqual(res.iloc[4], "fractales amis groupe s a s")
        self.assertEqual(res.iloc[5], "17560 ellis road tahlequah ok")
        self.assertEqual(res.iloc[6], "63 r de dieppe lille")
        self.assertEqual(res.iloc[7], "")
        self.assertEqual(res.iloc[8], "")
        self.assertEqual(res.iloc[9], "")

    def test_normalize_dataframe_preserves_original_columns(self):
        """Verify that original columns are never overwritten and normalized columns are created."""
        df_raw = pd.DataFrame({
            "entity_id": ["S1-1", "S1-2", "S1-3"],
            "business_name": ["Orelee's Barbershop", "राम मार्केटिंग", None],
            "business_address": ["1795 Westchester Dr", None, "63 R. DE DIEPPE"],
            "country": ["US", "India", "France"],
        })

        df_result = normalize_dataframe(df_raw, inplace=False)

        # Check original columns remain untouched
        self.assertEqual(df_result["business_name"].iloc[0], "Orelee's Barbershop")
        self.assertEqual(df_result["business_name"].iloc[1], "राम मार्केटिंग")
        self.assertTrue(pd.isna(df_result["business_name"].iloc[2]))
        self.assertEqual(df_result["country"].iloc[0], "US")
        self.assertEqual(df_result["country"].iloc[2], "France")

        # Check new normalized columns
        self.assertIn("business_name_norm", df_result.columns)
        self.assertIn("business_address_norm", df_result.columns)
        self.assertIn("country_norm", df_result.columns)

        self.assertEqual(df_result["business_name_norm"].iloc[0], "orelee s barbershop")
        self.assertEqual(df_result["business_name_norm"].iloc[1], "राम मार्केटिंग")
        self.assertEqual(df_result["business_name_norm"].iloc[2], "")
        self.assertEqual(df_result["country_norm"].iloc[0], "us")
        self.assertEqual(df_result["country_norm"].iloc[1], "india")
        self.assertEqual(df_result["country_norm"].iloc[2], "france")


if __name__ == "__main__":
    unittest.main()
