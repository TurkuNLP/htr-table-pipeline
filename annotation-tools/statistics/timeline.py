import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse

def extract_years_from_filename(filename):
    """
    Extracts years from filenames in the format 'parish_info_(start-year)-(end-year)_source.xml'.
    Returns the average of the two years if found, or None otherwise.
    """
    match = re.search(r'(\d{4})-(\d{4})', filename)
    if match:
        year_start, year_end = map(int, match.groups())
        return (year_start + year_end) / 2 
    return None

def analyze_years(directory):
    """
    Reads the files from the local cloned repository / directory and returns year-averages and year-distributions.
    """
    year_averages = []
    year_distribution = {}

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.xml'):
                avg_year = extract_years_from_filename(file)
                if avg_year is not None:
                    year_averages.append(avg_year)
                    rounded_year = int(round(avg_year))
                    year_distribution[rounded_year] = year_distribution.get(rounded_year, 0) + 1

    return year_averages, year_distribution

def plot_year_distribution(year_distribution):
    """
    Draws the plot showing annotation distribution over the years
    """
    years = sorted(year_distribution.keys())
    counts = [year_distribution[year] for year in years]

    plt.figure(figsize=(10, 6))
    plt.bar(years, counts, color='skyblue', edgecolor='black')
    plt.xlabel('Year')
    plt.ylabel('Number of Annotations')
    plt.title('Distribution of Annotations Over Time')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def main():
    directory = args.directory

    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    year_averages, year_distribution = analyze_years(directory)

    if year_averages:
        overall_average = np.mean(year_averages)
        print(f"Overall average year: {overall_average:.2f}")
    else:
        print("No valid year data found in the filenames.")
        return

    print("Plotting year distribution...")
    plot_year_distribution(year_distribution)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze XML file names for year averages and distributions.")
    parser.add_argument('--directory', type=str, help="Path to the cloned repository / directory containing XML files.")
    args = parser.parse_args()
    main()
