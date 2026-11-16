#!/usr/bin/env python3
"""Compare paper anchors against recorded outputs, without retraining."""
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2] / 'reference_results/scripts/results'
failures = 0
checks = 0


def compare(label, actual, expected):
    global checks, failures
    checks += 1
    precision = Decimal(expected).as_tuple().exponent
    rounded = Decimal(str(actual)).quantize(Decimal(1).scaleb(precision), rounding=ROUND_HALF_UP)
    passed = rounded == Decimal(expected)
    failures += not passed
    print(f"{'PASS' if passed else 'FAIL'} {label}: {actual} vs {expected}")


def read_rows(path):
    with (ROOT / path).open(newline='') as file:
        return list(csv.DictReader(file))


def metric(path, pattern):
    return re.search(pattern, (ROOT / path).read_text(), re.MULTILINE).group(1)


expected = {
    'amplitude_on': ('0.934', '0.005', '0.0069', '0.0009'),
    'probability_on': ('0.930', '0.003', '0.0069', '0.0012'),
    'hybrid_on': ('0.939', '0.005', '0.0062', '0.0006'),
    'amplitude_off': ('0.885', '0.003', '0.0207', '0.0006'),
    'probability_off': ('0.878', '0.007', '0.0246', '0.0007'),
    'hybrid_off': ('0.878', '0.007', '0.0246', '0.0007'),
}
encoding_rows = read_rows('final_gmm_diagW_pi/ablation_encoding_diagW/encoding_T20.csv')
for row in encoding_rows:
    for column, value in zip(('rf_mean', 'rf_std', 'mmd2_mean', 'mmd2_std'), expected[row['cell']]):
        compare(f"Table II {row['cell']} {column}", row[column], value)

for name, values in [('fixed24I', ('0.856', '0.0162', '9173')), ('diagW', ('0.941', '0.0049', '303'))]:
    rows = read_rows(f'final_gmm_diagW_pi/{name}/infer/summary.csv')
    row = next(entry for entry in rows if entry['T_infer'] == '20')
    for field, value in zip(('rf', 'mmd2', 'total_shots'), values):
        compare(f'Table I {name} {field}', row[field], value)

for field, value in [('rf', '0.945'), ('MMD2', '0.0033'), ('total_shots', '991')]:
    compare(f'Table I GMM 8q {field}', metric('final_8q/metrics_T20.txt', rf'adaptive\s*:.*?{field}=([0-9.]+)'), value)

for subdir, mmd, shots in [('3q/d8_diagW', '0.0312', '430'), ('3q/d16_diagW', '0.0259', '678'), ('8q', '0.0143', '993')]:
    path = f'final_image_paper/sec7_2_latent_{subdir}/metrics_T20_eta1.txt'
    compare(f'MNIST {subdir} MMD2', metric(path, r'adaptive\s*:.*?MMD2_pixel=([0-9.]+)'), mmd)
    compare(f'MNIST {subdir} shots', metric(path, r'adaptive\s*:.*?shots=([0-9.]+)'), shots)

for qubits, layers, gates in [(3, 12, 72), (8, 12, 192), (3, 20, 120), (8, 12, 192)]:
    compare(f'resource rotations {qubits}q/{layers}L', 2 * qubits * layers, str(gates))

for qubits, layers, gates in [(3, 12, 36), (8, 12, 96), (3, 20, 60), (8, 12, 96)]:
    compare(f'resource two-qubit gates {qubits}q/{layers}L', qubits * layers, str(gates))

baseline = int(metric('final_8q/metrics_T20.txt', r'high_shot:.*?total_shots=([0-9]+)'))
compare('high-shot baseline', baseline, '1600000')
for label, shots, expected_ratio in [('GMM 3q', 303, '5.3E+3'), ('GMM 8q', 991, '1.6E+3'),
                                     ('MNIST 3q', 430, '3.7E+3'), ('MNIST 4q', 678, '2.4E+3'),
                                     ('MNIST 8q', 993, '1.6E+3')]:
    compare(f'{label} approximate shot reduction', Decimal(baseline) / Decimal(shots), expected_ratio)

bounds = read_rows('final_gmm_diagW_pi/diagW/bounds/bounds_mmin_diagW.csv')
compare('lower bound cost ratio', Decimal(bounds[-1]['total_shots']) / Decimal(bounds[0]['total_shots']), '4.5')
compare('upper bound saturation at default', bounds[0]['cap_frac'], '0.000')
uniform = read_rows('final_gmm_diagW_pi/diagW/uniform_diagW.csv')
match = next(row for row in uniform if Decimal(row['rf']) >= Decimal('0.941'))
compare('uniform cost factor at matching rf', Decimal(match['total_shots']) / Decimal(303), '4')
eta = read_rows('final_gmm_diagW_pi/diagW/eta_diagW.csv')
cost = {row['eta']: Decimal(row['total_shots']) for row in eta}
compare('eta 0.1 versus 0.2 cost', cost['0.1'] / cost['0.2'], '3.8')
compare('eta 0 versus 1 cost order', cost['0.0'] / cost['1.0'], '1E+4')
encoding = {row['cell']: row for row in encoding_rows}
for name, value in [('amplitude', '3'), ('probability', '4'), ('hybrid', '4')]:
    off = Decimal(encoding[f'{name}_off']['mmd2_mean'])
    on = Decimal(encoding[f'{name}_on']['mmd2_mean'])
    compare(f'{name} time-off MMD factor', off / on, value)

print(f'{checks - failures}/{checks} comparisons passed; {failures} failed')
raise SystemExit(1 if failures else 0)
