"""
Тесты функции затрат по Downs & Vogel (1993), Table 9.
Запуск из корня проекта:  python tests/test_tep_economics.py   (или pytest tests)
"""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tep_economics import (BASE_CASE, calc_tep_operating_cost,
                           calc_tep_profit_extended, product_molar_flow)


class TestBaseCaseTable9(unittest.TestCase):
    """Пункты 1–3: контрольный расчёт базового режима."""

    def setUp(self):
        self.total, self.c = calc_tep_operating_cost(BASE_CASE)

    def test_total_170_6(self):
        self.assertAlmostEqual(self.total, 170.6, delta=0.15)

    def test_purge_loss_114_7(self):
        self.assertAlmostEqual(self.c['purge_value_per_kgmol'], 7.5973, delta=0.001)
        self.assertAlmostEqual(self.c['purge_loss'], 114.7, delta=0.1)

    def test_product_loss_30_3(self):
        self.assertAlmostEqual(self.c['product_loss_value_per_kgmol'], 0.1434, delta=0.0005)
        self.assertAlmostEqual(self.c['product_loss'], 30.3, delta=0.1)

    def test_compressor_18_3(self):
        self.assertAlmostEqual(self.c['compressor_cost'], 18.3, delta=0.05)

    def test_steam_7_3(self):
        self.assertAlmostEqual(self.c['steam_cost'], 7.3, delta=0.05)

    def test_product_molar_flow_211_3(self):
        # 9,21 кгмоль/м3 × 22,949 м3/ч ≈ 211,3 кгмоль/ч
        self.assertAlmostEqual(product_molar_flow(BASE_CASE), 211.3, delta=0.2)


class TestDimensionsAndZeros(unittest.TestCase):
    """Пункт 4: размерности и нулевые расходы."""

    def test_zero_flows_give_zero_cost(self):
        d = copy.deepcopy(BASE_CASE)
        for k in ('Purge Rate', 'Stripper Underflow', 'Stripper Steam Flow', 'Compressor Work'):
            d[k] = 0.0
        total, c = calc_tep_operating_cost(d)
        self.assertEqual(total, 0.0)

    def test_each_term_is_linear_in_its_flow(self):
        # удвоение расхода удваивает своё слагаемое и не трогает остальные
        _, c1 = calc_tep_operating_cost(BASE_CASE)
        for key, term in (('Purge Rate', 'purge_loss'), ('Stripper Underflow', 'product_loss'),
                          ('Compressor Work', 'compressor_cost'),
                          ('Stripper Steam Flow', 'steam_cost')):
            d = copy.deepcopy(BASE_CASE)
            d[key] *= 2
            _, c2 = calc_tep_operating_cost(d)
            self.assertAlmostEqual(c2[term], 2 * c1[term], places=9)
            self.assertAlmostEqual(c2['total_cost'] - c1['total_cost'], c1[term], places=9)

    def test_units_kgmol(self):
        # $/кгмоль × кгмоль/ч = $/ч: проверка через ручной расчёт продувки
        _, c = calc_tep_operating_cost(BASE_CASE)
        self.assertAlmostEqual(c['purge_molar_flow'], 44.79 * 0.33712, places=9)

    def test_xmeas_prefix_and_missing_key(self):
        pref = {'xmeas_' + k: v for k, v in BASE_CASE.items()}
        self.assertAlmostEqual(calc_tep_operating_cost(pref)[0],
                               calc_tep_operating_cost(BASE_CASE)[0], places=9)
        d = copy.deepcopy(BASE_CASE)
        del d['Compressor Work']
        with self.assertRaises(KeyError):
            calc_tep_operating_cost(d)


class TestProfitExtension(unittest.TestCase):
    """Пункт 5: выручка на вручную рассчитанном составе и расходе."""

    def test_revenue_manual(self):
        pG, pH = 30.44, 22.94    # пример цен; в работе — с отдельным обоснованием
        profit, c = calc_tep_profit_extended(BASE_CASE, pG, pH)
        F = 9.21 * 22.949
        manual = F * (0.53724 * pG + 0.43828 * pH)
        self.assertAlmostEqual(c['revenue'], manual, places=6)
        self.assertAlmostEqual(profit, manual - c['total_cost'], places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
