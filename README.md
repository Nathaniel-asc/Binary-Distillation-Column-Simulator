# Binary-Distillation-Column-Simulator

A small Python program that uses the McCabe–Thiele method to estimate theoretical stages for a binary distillation separation.

## Assumptions
- Constant relative volatility
- Saturated-liquid feed (`q = 1`)

## Inputs
- Relative volatility
- Feed composition
- Distillate purity
- Bottoms purity
- Reflux ratio

## Run
```bash
pip install -r requirements.txt
python simulator.py
```
