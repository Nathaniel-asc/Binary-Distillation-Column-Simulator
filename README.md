# Binary-Distillation-Column-Simulator

A small Python program that uses the McCabe–Thiele method to estimate theoretical stages for a binary distillation separation.

## Assumptions
- Constant relative volatility

## Inputs
- Relative volatility
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
