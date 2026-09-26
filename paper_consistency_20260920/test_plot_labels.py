"""Render the real plotting function and inspect editable text geometry."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import plot_boundary_fusion_fields as plot

class PlotLabelTest(unittest.TestCase):
    def test_method_labels_fit_reserved_gutter(self):
        tri = mtri.Triangulation([2, 17, 2, 17], [0, 0, 4, 4])
        u = np.array([1., 2., 1.5, 2.5])
        fields = {name: (u + offset, u * .1, u - 1.5 + offset)
                  for name, offset in [('truth', 0), ('s_only', .2), ('h_only', .1), ('t2c', .05)]}
        order = ['s_only', 'h_only', 't2c']
        metrics = {name: {'velocity_relative_l2': .01, 'pressure_relative_l2': .02} for name in order}
        with tempfile.TemporaryDirectory() as temp:
            args = argparse.Namespace(output_dir=Path(temp), png_only=True, dpi=70,
                plot_truth_pred=True, plot_error=True, xlim=(2,17), ylim=(0,4), error_cmap='magma')
            with patch.object(plt, 'close'):
                outputs = plot.plot_figures(args, tri, fields, order, 'test', metrics, {})
            self.assertEqual(len(outputs), 2)
            for number in plt.get_fignums():
                fig = plt.figure(number)
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                labels = [text for ax in fig.axes for text in ax.texts
                          if text.get_text() in plot.METHOD_ROW_LABELS.values()]
                self.assertGreaterEqual(len(labels), 3)
                for label in labels:
                    self.assertEqual(label.get_rotation(), 0, 'method labels must read horizontally')
                    self.assertNotIn('RA-LMoE', label.get_text())
                    box = label.get_window_extent(renderer)
                    self.assertGreaterEqual(box.x0, 0)
                    self.assertLessEqual(box.x1, label.axes.get_window_extent(renderer).x0)
                for first, second in zip(labels, labels[1:]):
                    self.assertFalse(first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer)))
        plt.close('all')

if __name__ == '__main__':
    unittest.main()
