import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import os

# --- Basic plotting style settings (inspired by plotting_tools.py and Figure 5 style) ---
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 20,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 14, # Adjusted for potentially longer labels
    'figure.figsize': (8, 6),
    'text.latex.preamble': r'\usepackage{txfonts} \usepackage{amsmath}',
    'pgf.rcfonts': False,
})

# --- Define which columns from the CSV contain algorithm ratios and their legend labels ---
# The keys are the column names in your CSV file.
# The values are how they will appear in the plot legend.
# This is based on the data generation script you provided.
ALGORITHMS_TO_PLOT = {
    "ratio_baseline2": "Baseline2 Density Ratio (DP)",
    "ratio_our": "Our Method Density Ratio (LDP)",
    "ratio_baseline3": "Simple_LEDP Density Ratio",
}

# --- (Optional) Define specific styles for these algorithms ---
# Keys here MUST match the legend labels defined in ALGORITHMS_TO_PLOT.
# If an algorithm is not in this dictionary, default styles will be used.
ALGO_STYLES = {
    'Baseline2 Density Ratio (DP)': {'color': 'tab:blue', 'marker': 's', 'linestyle': '--'},
    'Our Method Density Ratio (LDP)': {'color': 'tab:red', 'marker': 'o', 'linestyle': '-'},
    'Simple_LEDP Density Ratio': {'color': 'tab:green', 'marker': '^', 'linestyle': ':'},
    # Add more styles if ALGORITHMS_TO_PLOT contains other algorithms
}

# Default styles for algorithms not in ALGO_STYLES
DEFAULT_COLORS = plt.cm.viridis(np.linspace(0, 0.9, 7))
DEFAULT_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'H']
DEFAULT_LINESTYLES = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1))]


def plot_density_ratio_vs_epsilon(csv_filepath, dataset_name_for_title, output_filename,
                                  epsilon_col_name="epsilon", algorithms_map=None):
    """
    Generates a "Density Ratio vs. Epsilon" plot from a CSV file in wide format.

    Args:
        csv_filepath (str): Path to the input CSV file.
        dataset_name_for_title (str): Name of the dataset for the plot title.
        output_filename (str): Path to save the output plot.
        epsilon_col_name (str): Name of the column containing epsilon values.
        algorithms_map (dict): Dictionary mapping CSV column names for ratios
                               to their desired legend labels.
    """
    if algorithms_map is None:
        algorithms_map = ALGORITHMS_TO_PLOT

    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}")
        return
    except Exception as e:
        print(f"Error reading CSV file {csv_filepath}: {e}")
        return

    if not isinstance(df, pd.DataFrame) or df.empty:
        print(f"Error: No data loaded from {csv_filepath}. The DataFrame is empty.")
        return

    # Check if epsilon column exists
    if epsilon_col_name not in df.columns:
        print(f"Error: Epsilon column '{epsilon_col_name}' not found in {csv_filepath}.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    fig, ax = plt.subplots()

    # Check for missing algorithm ratio columns
    missing_algo_cols = [col for col in algorithms_map.keys() if col not in df.columns]
    if len(missing_algo_cols) == len(algorithms_map):
        print(f"Error: None of the specified algorithm ratio columns {list(algorithms_map.keys())} were found in {csv_filepath}.")
        print(f"Available columns: {df.columns.tolist()}")
        plt.close(fig) # Close the empty figure
        return
    elif missing_algo_cols:
        print(f"Warning: The following algorithm ratio columns were not found in {csv_filepath} and will be skipped: {missing_algo_cols}")


    algo_style_keys_lower = {k.lower(): k for k in ALGO_STYLES.keys()}

    plot_idx = 0
    for ratio_col, legend_label in algorithms_map.items():
        if ratio_col not in df.columns:
            continue # Skip if this specific ratio column is missing

        # Group by epsilon and calculate mean ratio (handles multiple runs)
        # Ensure the ratio_col is numeric before aggregation
        if not pd.api.types.is_numeric_dtype(df[ratio_col]):
            print(f"Warning: Ratio column '{ratio_col}' is not numeric in {csv_filepath}. Skipping this algorithm.")
            continue

        # Drop rows where epsilon or ratio_col is NaN to avoid issues with groupby/mean
        # This is important if data can be incomplete.
        valid_df = df[[epsilon_col_name, ratio_col]].dropna()
        if valid_df.empty:
            print(f"Warning: No valid data for epsilon '{epsilon_col_name}' and ratio '{ratio_col}' after dropping NaNs. Skipping.")
            continue

        mean_ratios_df = valid_df.groupby(epsilon_col_name)[[ratio_col]].mean().reset_index()
        mean_ratios_df = mean_ratios_df.sort_values(by=epsilon_col_name)

        if mean_ratios_df.empty:
            print(f"Warning: No data to plot for algorithm '{legend_label}' after processing. Skipping.")
            continue

        # Get style for 'legend_label' from ALGO_STYLES or use default
        style_key_lookup = legend_label.lower()
        actual_style_key = algo_style_keys_lower.get(style_key_lookup)
        if not actual_style_key and legend_label in ALGO_STYLES: # Try exact match if lowercased failed
            actual_style_key = legend_label

        specific_style = ALGO_STYLES.get(actual_style_key, {}) if actual_style_key else {}


        color = specific_style.get('color', DEFAULT_COLORS[plot_idx % len(DEFAULT_COLORS)])
        marker = specific_style.get('marker', DEFAULT_MARKERS[plot_idx % len(DEFAULT_MARKERS)])
        linestyle = specific_style.get('linestyle', DEFAULT_LINESTYLES[plot_idx % len(DEFAULT_LINESTYLES)])

        ax.plot(mean_ratios_df[epsilon_col_name], mean_ratios_df[ratio_col],
                label=legend_label,
                marker=marker,
                linestyle=linestyle,
                color=color,
                linewidth=2,
                markersize=7)
        plot_idx += 1

    # Horizontal line at y=1.0
    ax.axhline(1.0, linestyle='--', color='gray', linewidth=1.5, label='Non-Private (Optimal Ratio)')

    # Labels and Title
    ax.set_xlabel(r'$\epsilon$')
    ax.set_ylabel('Ratio of Private to True Density')
    ax.set_title(dataset_name_for_title)

    # Scale and Ticks for Y-axis (Log Scale)
    ax.set_yscale('log')
    y_ticks_fig5_style = [0.1, 0.2, 0.5, 1.0] # Key ticks from Figure 5
    ax.set_yticks(y_ticks_fig5_style)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    # Determine Y-axis limits
    min_ratio_data = float('inf')
    max_ratio_data = float('-inf')
    for ratio_col in algorithms_map.keys():
        if ratio_col in df.columns and pd.api.types.is_numeric_dtype(df[ratio_col]):
            # Consider only non-NaN values for min/max calculation
            valid_ratios = df[ratio_col].dropna()
            if not valid_ratios.empty:
                min_ratio_data = min(min_ratio_data, valid_ratios.min())
                max_ratio_data = max(max_ratio_data, valid_ratios.max())

    if min_ratio_data == float('inf'): min_ratio_data = 0.01 # Default if no ratio data
    if max_ratio_data == float('-inf'): max_ratio_data = 1.5 # Default if no ratio data


    ylim_bottom = min(0.05, min_ratio_data * 0.8) if min_ratio_data > 0 else 0.01
    if ylim_bottom <= 0: ylim_bottom = 0.005 # Ensure log scale has positive lower limit, slightly lower for flexibility

    ylim_top = 1.5
    if max_ratio_data > 1.2: # If data significantly exceeds 1.2
        ylim_top = max(max_ratio_data * 1.1, 1.5)
    ax.set_ylim(ylim_bottom, ylim_top)


    # X-axis Ticks and Limits (adaptive based on epsilon range)
    all_epsilons_sorted = np.sort(df[epsilon_col_name].dropna().unique())
    if len(all_epsilons_sorted) > 0:
        min_epsilon_data = all_epsilons_sorted[0]
        max_epsilon_data = all_epsilons_sorted[-1]

        # Heuristics for x-axis ticks, similar to previous version
        if max_epsilon_data <= 1.05 and any(e in all_epsilons_sorted for e in [0.1, 0.25, 0.5, 1.0]):
            custom_ticks_x_set = {0.1, 0.25, 0.5, 1.0}
            custom_ticks_x = sorted(list(set(e for e in all_epsilons_sorted if e in custom_ticks_x_set) | custom_ticks_x_set))
            custom_ticks_x = [t for t in custom_ticks_x if t >= min_epsilon_data*0.8 and t <= max_epsilon_data*1.2 + 0.05]
            if not custom_ticks_x and len(all_epsilons_sorted) > 0: custom_ticks_x = np.unique([round(e,2) for e in all_epsilons_sorted])
            elif not custom_ticks_x: custom_ticks_x = [0.1, 0.5, 1.0]
            ax.set_xticks(sorted(list(set(custom_ticks_x))))
            ax.set_xlim(max(0, min_epsilon_data - 0.1 if min_epsilon_data > 0.1 else 0), min(1.05, max_epsilon_data + 0.1 if max_epsilon_data < 1 else 1.05) )
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))

        elif max_epsilon_data > 1.0 and max_epsilon_data <= 10.0: # E.g. for Central DP epsilons up to 8 or 10
            step = 1.0 if (max_epsilon_data - min_epsilon_data) / 1.0 < 10 else 2.0
            start_tick = np.floor(min_epsilon_data / step) * step
            end_tick = np.ceil(max_epsilon_data / step) * step + step / 2
            custom_ticks_x = np.arange(start_tick, end_tick, step)
            if min_epsilon_data < step and 0 not in custom_ticks_x and 0 >= start_tick:
                custom_ticks_x = np.insert(custom_ticks_x[custom_ticks_x > 0], 0, 0)
            custom_ticks_x = sorted(list(set(custom_ticks_x)))
            ax.set_xticks(custom_ticks_x)
            ax.set_xlim(max(-0.2, min_epsilon_data - step*0.2), max_epsilon_data + step*0.2)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        else: # Generic fallback
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both', integer=(max_epsilon_data - min_epsilon_data < 10 and min_epsilon_data >=0)))
            ax.set_xlim(min_epsilon_data * 0.9 if min_epsilon_data > 0 else -0.05 * max_epsilon_data, max_epsilon_data * 1.05)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))
    else:
        print(f"Warning: No valid epsilon values found in column '{epsilon_col_name}'. X-axis may be incorrect.")
        ax.set_xticks([0,1]) # Default ticks if no epsilon data

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if handles: # Only show legend if there are items to show
        if len(all_epsilons_sorted)>0 and max_epsilon_data <= 1.05:
            ax.legend(loc='upper left')
        elif len(all_epsilons_sorted)>0 and max_epsilon_data > 5 and (min_ratio_data != float('inf') and min_ratio_data < 0.8):
            ax.legend(loc='lower right')
        else:
            ax.legend(loc='best')

    # Grid
    ax.grid(True, which="major", axis='y', linestyle=':', linewidth=0.5, alpha=0.7)
    ax.grid(True, which="major", axis='x', linestyle=':', linewidth=0.5, alpha=0.7)

    # Save plot
    plt.tight_layout()
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close(fig) # Close figure to free memory

# --- Main part of the script to call the plotting function for each dataset ---
if __name__ == "__main__":
    base_data_dir = "untxt12"  # Directory where your CSV files are located

    # Define datasets. Assumes all these CSVs follow the "wide" format
    # generated by your script.
    datasets_info = [
        {"name_for_title": "Squirrel", "short_name": "Squirrel", "path_suffix": "squirreldetailed_density_ratios_all_runs.csv"},
        {"name_for_title": "DE", "short_name": "DE", "path_suffix": "DEdetailed_density_ratios_all_runs.csv"},
        {"name_for_title": "Facebook", "short_name": "Facebook", "path_suffix": "facebookdetailed_density_ratios_all_runs.csv"},
        {"name_for_title": "GRQC", "short_name": "GRQC", "path_suffix": "GRQCdetailed_density_ratios_all_runs.csv"},
        {"name_for_title": "ENGB", "short_name": "ENGB", "path_suffix": "ENGBdetailed_density_ratios_all_runs.csv"},
    ]

    output_dir = "plots_from_wide_format"  # Directory to save the plots
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    for dataset in datasets_info:
        csv_file_path = os.path.join(base_data_dir, dataset["path_suffix"])
        output_plot_file = os.path.join(output_dir, f"{dataset['short_name']}_density_ratio_plot.pdf") # Save as PDF

        # The title can be customized further if needed, e.g., adding (LEDP) or (Central DP)
        # title = f"{dataset['name_for_title']} (Type of DP, $\delta=...$)"
        title = dataset['name_for_title']

        # Call the plotting function
        # It will use the global ALGORITHMS_TO_PLOT and ALGO_STYLES
        plot_density_ratio_vs_epsilon(csv_file_path, title, output_plot_file)