 Pricing-Pattern-Shift-in-Real-Estate-Post-Pandemic
- Overview

This project analyzes how real estate pricing patterns have shifted since the COVID-19 pandemic. Using open real-estate listing data, it models monthly median price trends, compares pre-pandemic vs post-pandemic values, and generates visual insights into how property markets evolved over time.

The entire pipeline is written in Python with minimal dependencies, making it easy to run, extend, or deploy.

Key Features

1. Timeline-based trend analysis (2017–2024 or your dataset range)
2. Grouping by city / location / property type
3. Pandemic shock detection (2020–2021 dip)
4. Post-pandemic recovery comparison
5. Pre-vs-Post % change statistics
6. Automatic visualization (PNG)
7. CSV exports of trends + summary stats
8. Optional forecasting model (linear regression, only if scikit-learn is installed)
9. Works even with incomplete CSVs thanks to auto-column detection
10. Synthetic dataset generator if you don't provide real data



How to Run the Project

You can run this project with **your own real-estate dataset** or use the **built-in synthetic data generator**.

Run with Your CSV Data

Replace `yourfile.csv` with the path to your dataset:

bash
python pricing_pattern_shift.py --data yourfile.csv --output results


Example (Windows):

bash
python pricing_pattern_shift.py --data "C:\Users\ragha\Downloads\Bengaluru_House_Data.csv" --output results

All output files (charts + CSV summaries) will be saved to the results/ folder.

- Run With Synthetic Data (No CSV Needed)

If you don’t provide a dataset, the script auto-generates a realistic synthetic real-estate dataset and runs the full pipeline:

bash
python pricing_pattern_shift.py


-  Save Outputs to a Custom Folder
You can specify your own output directory:

bash
python pricing_pattern_shift.py --data yourfile.csv --output analysis_output

The folder will be created automatically.

- View Help / Usage Instructions

bash
python pricing_pattern_shift.py --help

