from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic price data.")
    parser.add_argument("--rows", type=int, default=500, help="Number of rows to generate.")
    parser.add_argument("--out", required=True, help="Path to write CSV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    size_sqft = rng.normal(1500, 350, size=args.rows).clip(400, 4000)
    bedrooms = rng.integers(1, 6, size=args.rows)
    bathrooms = rng.integers(1, 4, size=args.rows)
    age_years = rng.integers(0, 50, size=args.rows)
    distance_km = rng.normal(8, 4, size=args.rows).clip(0.5, 40)

    base_price = 50000
    price = (
        base_price
        + size_sqft * 220
        + bedrooms * 12000
        + bathrooms * 9000
        - age_years * 1000
        - distance_km * 1500
        + rng.normal(0, 20000, size=args.rows)
    )

    df = pd.DataFrame(
        {
            "size_sqft": size_sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age_years,
            "distance_km": distance_km,
            "price": price.round(2),
        }
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Sample data saved to: {out_path}")


if __name__ == "__main__":
    main()
