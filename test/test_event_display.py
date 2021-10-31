import unittest
import sys
import os

sys.path.append("../")
import event_display

from lxml import etree as ET
from event_display import get_module_geometry, get_module_positions, plot_calorimeter_hits


class TestEventDisplay(unittest.TestCase):

    def setUp(self) -> None:
        self.geometry_xml = ET.parse(f"{os.path.dirname(__file__)}/geometry/athena.gdml")
        return super().setUp()

    def test_load_modules(self):

        positions = get_module_positions('crystal_module', self.geometry_xml)
        self.assertTrue(positions)
        ids = list(positions.keys())
        self.assertEqual(0x55600cea9670, ids[0])

    def test_get_module_geometry(self):
        pwo_size_x, pwo_size_y, pwo_size_z, unit = get_module_geometry('crystal_box', self.geometry_xml)
        self.assertTrue(pwo_size_x)
        self.assertTrue(pwo_size_y)
        self.assertTrue(pwo_size_z)
        self.assertTrue(unit)


if __name__ == '__main__':
    unittest.main()

