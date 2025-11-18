"""
Pricing Pattern Shift in Real Estate — Data Science Project
Single-file Python project (minimal dependencies) for analyzing how real estate prices
changed after the COVID-19 pandemic using open listing data.

Usage:
    python pricing_pattern_shift.py --data path/to/listings.csv --output results/

If no data file is provided the script will generate a small synthetic dataset so you
can run the pipeline end-to-end.

Minimal dependencies (recommended):
    pandas >= 1.1
    numpy
    matplotlib
    scikit-learn  (optional, used only for an optional predictive model)

Design goals:
 - single script, modular functions
 - robust to missing columns (works with common listing CSVs that contain: date, price, area, city/neighborhood, property_type)
 - light-weight visualizations using matplotlib
 - optional modeling with scikit-learn guarded by try/except

Author: Generated for a Data Science project
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import sklearn but keep it optional
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ------------------------- Data utilities -------------------------

def load_listings(path):
    """Load CSV into DataFrame. Attempts to coerce common column names.

    Expected useful columns (any subset):
      - date (listing or transaction date)
      - price (numeric)
      - area (square feet / square meter)
      - city / neighborhood / location
      - property_type (house/apartment)

    Returns: pandas.DataFrame
    """
    df = pd.read_csv(path)

    # normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # guess columns
    possible_date = [c for c in df.columns if 'date' in c]
    possible_price = [c for c in df.columns if 'price' in c]
    possible_area = [c for c in df.columns if c in ('area','sqft','sqm','size','builtup','size_sqft') or 'area' in c or 'sqft' in c or 'size' in c]
    possible_city = [c for c in df.columns if c in ('city','town','location','neighborhood','state')]
    possible_type = [c for c in df.columns if 'property' in c or 'type' in c]

    # rename to standard names
    col_map = {}
    if possible_date:
        col_map[possible_date[0]] = 'date'
    if possible_price:
        col_map[possible_price[0]] = 'price'
    if possible_area:
        col_map[possible_area[0]] = 'area'
    if possible_city:
        col_map[possible_city[0]] = 'location'
    if possible_type:
        col_map[possible_type[0]] = 'property_type'

    df = df.rename(columns=col_map)

    return df


def generate_synthetic_listings(n=1000, seed=42):
    """Create a small synthetic dataset that mimics pre- and post-pandemic pricing patterns.
    This is useful to test the pipeline without real listings.
    """
    rng = np.random.RandomState(seed)
    # Dates from 2017-01-01 to 2024-12-31
    start = datetime(2017, 1, 1)
    end = datetime(2024, 12, 31)
    days = (end - start).days
    dates = [start + pd.Timedelta(days=int(x)) for x in rng.uniform(0, days, size=n)]

    # locations
    locations = ['MetroCity', 'SuburbA', 'SuburbB']
    property_types = ['apartment', 'house']

    # base price per sqft (set different trends for locations)
    base = {
        'MetroCity': 2000,
        'SuburbA': 1200,
        'SuburbB': 900
    }

    rows = []
    for d in dates:
        loc = rng.choice(locations)
        ptype = rng.choice(property_types, p=[0.7, 0.3])
        # area
        area = max(300, int(rng.normal(900, 300)))

        # simulate price trend: small growth pre-2020, dip/shift in 2020-2021, then divergent recoveries
        year = d.year + (d.timetuple().tm_yday / 365.0)
        # pre-2020 growth
        trend = 1 + 0.03 * (year - 2017)
        # pandemic effect
        if year >= 2020 and year < 2022:
            # temporary shock + location-specific effect
            shock = rng.normal(0.9, 0.05)  # 10% drop on average
            trend = trend * shock
        # post-2022 divergent growth (metro slower, suburbs faster)
        if year >= 2022:
            if loc == 'MetroCity':
                trend *= 1 + 0.02 * (year - 2022)
            else:
                trend *= 1 + 0.05 * (year - 2022)

        price_per_sqft = base[loc] * trend
        price = price_per_sqft * area * (1 + rng.normal(0, 0.08))

        rows.append({'date': d.strftime('%Y-%m-%d'), 'price': price, 'area': area, 'location': loc, 'property_type': ptype})

    df = pd.DataFrame(rows)
    return df


# ------------------------- Preprocessing -------------------------

def preprocess(df):
    """Standardize and clean the DataFrame. Adds price_per_unit and datetime index.

    Returns cleaned DataFrame.
    """
    df = df.copy()

    # parse date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        # if no date, create synthetic increasing index (not ideal)
        df['date'] = pd.to_datetime('2017-01-01') + pd.to_timedelta(np.arange(len(df)), unit='D')

    # coerce price and area
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
    else:
        df['price'] = np.nan

    if 'area' in df.columns:
        df['area'] = pd.to_numeric(df['area'], errors='coerce')
    else:
        df['area'] = np.nan

    # price per unit
    df['price_per_area'] = df['price'] / df['area']

    # drop obviously broken rows
    df = df[df['price'] > 1000]  # filter tiny prices
    df = df[df['area'] > 50]

    # fill location/type
    if 'location' not in df.columns:
        df['location'] = 'unknown'
    if 'property_type' not in df.columns:
        df['property_type'] = 'unknown'

    # set date index
    df = df.sort_values('date').set_index('date')

    return df


# ------------------------- Analysis functions -------------------------

def monthly_median_trend(df, group_by='location'):
    """Compute monthly median price_per_area trends for each group (default location).

    Returns a DataFrame with a datetime index (month start) and columns for each group.
    """
    df2 = df.copy()
    df2['month'] = df2.index.to_period('M').to_timestamp()
    grouped = df2.groupby(['month', group_by])['price_per_area'].median().reset_index()
    pivot = grouped.pivot(index='month', columns=group_by, values='price_per_area')
    return pivot


def plot_trends(pivot, title='Monthly median price per area'):
    """Simple matplotlib plot. Saves figure files in current directory.
    """
    plt.figure(figsize=(10, 5))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], label=str(col))
    plt.title(title)
    plt.xlabel('Month')
    plt.ylabel('Median price per unit area')
    plt.legend()
    plt.tight_layout()
    fname = 'trend_monthly_median.png'
    plt.savefig(fname)
    print('Saved trend chart to', fname)
    plt.close()


def compute_change_stats(pivot, baseline_start='2017-01-01', baseline_end='2019-12-31', post_start='2022-01-01', post_end='2024-12-31'):
    """Compute percent change of median price_per_area from baseline period to post-pandemic period for every group.

    Returns a DataFrame with columns: baseline_median, post_median, pct_change
    """
    # convert strings to timestamps
    bs = pd.to_datetime(baseline_start)
    be = pd.to_datetime(baseline_end)
    ps = pd.to_datetime(post_start)
    pe = pd.to_datetime(post_end)

    baseline = pivot[(pivot.index >= bs) & (pivot.index <= be)].median()
    post = pivot[(pivot.index >= ps) & (pivot.index <= pe)].median()

    df = pd.DataFrame({'baseline_median': baseline, 'post_median': post})
    df['pct_change'] = (df['post_median'] - df['baseline_median']) / df['baseline_median'] * 100
    return df.sort_values('pct_change', ascending=False)


# ------------------------- Simple predictive model (optional) -------------------------

def train_simple_trend_model(pivot, group, months_ahead=6):
    """Train a tiny linear regression to forecast future monthly median price for one group.
    Requires sklearn. If sklearn is not available the function will raise RuntimeError.
    This is intentionally simple: linear model on time index.
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError('scikit-learn not available. Install scikit-learn to enable modeling.')

    series = pivot[group].dropna()
    if len(series) < 12:
        raise ValueError('Not enough historical months to train model.')

    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.arange(len(series), len(series) + months_ahead).reshape(-1, 1)
    preds = model.predict(future_X).ravel()

    # return forecast index and values
    last_month = series.index[-1]
    future_index = pd.date_range(last_month + pd.offsets.MonthBegin(1), periods=months_ahead, freq='MS')
    forecast = pd.Series(preds, index=future_index)
    return model, forecast


# ------------------------- Main pipeline -------------------------

def run_pipeline(data_path=None, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)

    if data_path is None:
        print('No data provided — generating synthetic dataset to demonstrate the pipeline.')
        df = generate_synthetic_listings(n=1500)
    else:
        print('Loading data from', data_path)
        df = load_listings(data_path)

    df = preprocess(df)
    print('Data after cleaning: ', df.shape)

    pivot = monthly_median_trend(df, group_by='location')

    # quick diagnostics: save pivot
    pivot.to_csv(os.path.join(output_dir, 'monthly_median_by_location.csv'))

    plot_trends(pivot, title='Monthly median price per area by location')

    stats = compute_change_stats(pivot)
    stats.to_csv(os.path.join(output_dir, 'pre_post_change_stats.csv'))
    print('\nPercent change from baseline to post-pandemic (saved to pre_post_change_stats.csv):\n')
    print(stats)

    # run optional model for the top location
    try:
        top_location = stats.index[0]
        if SKLEARN_AVAILABLE:
            model, forecast = train_simple_trend_model(pivot, top_location, months_ahead=12)
            forecast.to_csv(os.path.join(output_dir, f'forecast_{top_location}.csv'))
            print('\nSaved simple linear forecast for', top_location)
        else:
            print('\nscikit-learn not available — skipping modeling step. To enable, install scikit-learn.')
    except Exception as e:
        print('\nModeling step skipped due to:', str(e))

    print('\nPipeline completed. Check the', output_dir, 'folder for outputs (CSV + PNG).')


# ------------------------- CLI -------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pricing Pattern Shift in Real Estate — lightweight pipeline')
    parser.add_argument('--data', help='Path to CSV file with listings data (optional). If omitted, synthetic data is used.')
    parser.add_argument('--output', default='results', help='Output folder (default: results)')
    args = parser.parse_args()

    run_pipeline(data_path=args.data, output_dir=args.output)
