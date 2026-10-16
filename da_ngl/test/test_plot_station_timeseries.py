"""Physical-unit and metric checks for the station time-series plotter."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'main_code'))
from plot_station_timeseries import metric_rows, physical_variables, variable_unit


class StationTimeseriesTests(unittest.TestCase):
    def test_physical_units_and_wind_derived_after_denormalization(self):
        channels = ['msl'] + [f'{v}{p}' for v in ('t', 'r', 'u', 'v', 'z')
                              for p in (850, 700, 500)]
        mean = np.array([100000.] + [v for v in (273.15, 50., 3., 4., 9806.65)
                                    for _ in range(3)])
        data = physical_variables(np.zeros((2, 5, len(channels))), channels,
                                  mean, np.ones(len(channels)))
        for name, expected in [('msl', 1000), ('t850', 0), ('r700', 50),
                               ('wind850', 5), ('z500', 1000)]:
            np.testing.assert_allclose(data[name], expected, atol=1e-10)
        self.assertEqual(data['msl'].shape, (2, 5))

    def test_metrics_are_per_station_and_time_averaged(self):
        values = {'ERA5': {'msl': np.array([[1000.], [1002.]])},
                  'FuXi-24h': {'msl': np.array([[1002.], [1004.]])},
                  'Analysis': {'msl': np.array([[1001.], [1001.]])}}
        row = metric_rows(values, [{'station_id': 'TEST'}]).iloc[0]
        self.assertEqual(row.fuxi_mae, 2)
        self.assertEqual(row.analysis_mae, 1)
        self.assertEqual(row.analysis_rmse, 1)
        self.assertEqual(row.analysis_bias, 0)
        self.assertEqual(row.mae_improvement_pct, 50)
        self.assertEqual(row.n_times, 2)

    def test_zero_background_error_does_not_report_infinite_improvement(self):
        values = {source: {'t850': np.ones((2, 1))}
                  for source in ('ERA5', 'FuXi-24h', 'Analysis')}
        row = metric_rows(values, [{'station_id': 'TEST'}]).iloc[0]
        self.assertTrue(np.isnan(row.mae_improvement_pct))
        self.assertEqual(variable_unit('z700'), 'gpm')
        self.assertEqual(variable_unit('wind700'), 'm/s')


if __name__ == '__main__':
    unittest.main()
