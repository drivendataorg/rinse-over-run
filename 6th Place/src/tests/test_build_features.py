#!/usr/bin/env python
import pandas as pd
from src.features.build_features import calc_number_of_skipped_phases


def test_calc_number_of_skipped_phases():
    for exp, obs, answers in [
        ('1111', '1111', [0, 0, 0, 0]),
        ('1111', '1???', [0, 0, 0, 0]),
        ('1111', '1010', [0, 1, 1, 2])
    ]:
        df = pd.DataFrame([{
            'expected_phase_summary': exp,
            'phase_summary': obs
        }])
        row = calc_number_of_skipped_phases(df).iloc[0]
        for p, exp_answer in zip(range(1, 5), answers):
            assert row[f'f_phase_{p}_number_of_skipped_phases'] == exp_answer
