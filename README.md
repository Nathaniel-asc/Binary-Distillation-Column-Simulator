# Binary-Distillation-Column-Simulator

A small Python program that uses the McCabe–Thiele method to estimate theoretical stages for a binary distillation separation.

## VLE input options

- Constant relative volatility, using the entered `alpha` value.
- Direct VLE data from a CSV file. Choose option `2` at the prompt and provide a CSV path.

CSV files must have header columns named `x` and `y`, with one equilibrium point per row. Values must be finite mole fractions between 0 and 1, strictly increasing in both columns, and include the endpoints `(0, 0)` and `(1, 1)`.

The program also detects interior intersections with `y = x` (azeotropes). An intersection listed in the CSV is reported exactly; one between points is estimated by linear interpolation. For a feed below an azeotrope in the light-key region, `xD` must remain below the azeotrope; product specifications on the other side are rejected and re-prompted. `xB` is likewise kept above any lower azeotropic boundary.

```csv
x,y
0.0,0.0
0.2,0.43
0.5,0.70
0.8,0.91
1.0,1.0
```

See `example_vle_data.csv` for a regular VLE curve and `example_azeotropic_vle_data.csv` for a curve with an azeotrope at `x = y = 0.5`.
- Constant Molar Overflow (CMO)
- Negligible heat effects
- Isobaric operation

## Assumptions
- Constant Molar Overflow (CMO)
- Negligible heat effects
- Isobaric operation

## Inputs
- Relative volatility or VLE data
- Feed composition
- Distillate purity
- Bottoms purity
- Feed thermal condition (`q`)
- Reflux ratio

The reported value is rounded up to a whole number of theoretical stages when the final reboiler stage is partial. A total condenser is not counted as an equilibrium stage.

## Run
```bash
pip install -r requirements.txt
python simulator.py
```
