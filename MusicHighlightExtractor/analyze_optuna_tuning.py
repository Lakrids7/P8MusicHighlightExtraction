# analyze_study.py
import optuna
import pandas as pd
import plotly.io as pio # For saving plots if 'show()' doesn't work in your environment

# --- Configuration ---
STUDY_NAME = "HuangOptunaTuning"  # MUST match the name used in your training script
STORAGE_URL = "sqlite:///huang_optuna_tuning.db" # MUST match the database file used

# --- Set Pandas display options for better readability ---
pd.set_option('display.max_rows', 200)      # Show more rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)        # Wider display for columns
pd.set_option('display.max_colwidth', 100)  # Show more of long string content

def analyze_optuna_study(study_name: str, storage_url: str):
    """Loads an Optuna study and prints summaries and generates plots."""
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except Exception as e:
        print(f"Error loading study '{study_name}' from '{storage_url}': {e}")
        print("Please ensure the study name and storage URL are correct and the database file exists.")
        return

    print(f"\n--- Optuna Study Summary: '{study.study_name}' ---")
    print(f"Number of trials in study: {len(study.trials)}")

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Number of COMPLETED trials: {len(completed_trials)}")

    if not completed_trials:
        print("No trials have completed successfully. Cannot provide detailed analysis.")
        return

    # --- 1. Best Trial Information ---
    if study.best_trial:
        print("\n--- Best Trial ---")
        print(f"  Trial Number: {study.best_trial.number}")
        print(f"  Objective Value (Loss): {study.best_trial.value:.6f}")
        print("  Hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
        if "val_mae_sec" in study.best_trial.user_attrs:
             print(f"  Validation MAE (sec): {study.best_trial.user_attrs['val_mae_sec']:.3f}")
        else:
            print("  Validation MAE (sec): Not found in user_attrs")
    else:
        print("\nNo best trial found (e.g., all trials failed or were pruned without reporting a value).")

    # --- 2. Pandas DataFrame Summary ---
    print("\n--- All Trials Summary (Sorted by Objective Value) ---")
    try:
        df_trials = study.trials_dataframe()
        # Define columns to display for brevity and relevance
        param_cols = [col for col in df_trials.columns if col.startswith('params_')]
        user_attr_cols = [col for col in df_trials.columns if col.startswith('user_attrs_')]
        # Ensure 'value' (loss) and 'user_attrs_val_mae_sec' are present if they exist
        metric_cols = []
        if 'value' in df_trials.columns: metric_cols.append('value')
        if 'user_attrs_val_mae_sec' in df_trials.columns: metric_cols.append('user_attrs_val_mae_sec')


        display_cols = ['number', 'state', 'duration'] + metric_cols + param_cols
        if 'user_attrs_val_mae_sec' not in display_cols and 'user_attrs_val_mae_sec' in df_trials.columns: # ensure it's added if missed
            display_cols.append('user_attrs_val_mae_sec')


        # Filter out columns that don't exist in the dataframe to prevent errors
        display_cols = [col for col in display_cols if col in df_trials.columns]

        if 'value' in df_trials.columns:
            print(df_trials[display_cols].sort_values(by='value', ascending=True).head(20)) # Show top 20
        else:
            print("DataFrame does not contain 'value' column for sorting.")
            print(df_trials[display_cols].head(20))

    except Exception as e:
        print(f"Error generating DataFrame summary: {e}")

    # --- 3. Generate and Save/Show Visualization Plots ---
    # These require plotly and matplotlib to be installed in your environment
    # If running in a headless environment, .show() might not work; .write_html() is better.

    plot_dir = "optuna_plots"
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nSaving plots to directory: {plot_dir}")

    # Plot 1: Optimization History
    if len(completed_trials) > 0:
        try:
            fig_history = optuna.visualization.plot_optimization_history(study)
            # fig_history.show()
            fig_history.write_html(os.path.join(plot_dir, "optimization_history.html"))
            print(f"  Saved: {os.path.join(plot_dir, 'optimization_history.html')}")
        except Exception as e:
            print(f"  Skipping optimization_history plot: {e}")

    # Plot 2: Parameter Importances
    if len(completed_trials) >= 2: # Needs at least 2 completed trials
        try:
            fig_importance = optuna.visualization.plot_param_importances(study)
            # fig_importance.show()
            fig_importance.write_html(os.path.join(plot_dir, "param_importances.html"))
            print(f"  Saved: {os.path.join(plot_dir, 'param_importances.html')}")
        except Exception as e:
            print(f"  Skipping param_importances plot: {e} (Often needs more trials or successful value reporting)")

    # Plot 3: Parallel Coordinate
    if len(completed_trials) > 0:
        try:
            fig_parallel = optuna.visualization.plot_parallel_coordinate(study)
            # fig_parallel.show()
            fig_parallel.write_html(os.path.join(plot_dir, "parallel_coordinate.html"))
            print(f"  Saved: {os.path.join(plot_dir, 'parallel_coordinate.html')}")
        except Exception as e:
            print(f"  Skipping parallel_coordinate plot: {e}")

    # Plot 4: Slice Plot (shows objective value against individual hyperparameters)
    if len(completed_trials) > 0 and study.best_trial: # Check if best_trial exists to get params
        try:
            # study.best_params might be empty if no trial completed successfully with a value
            params_to_slice = list(study.best_trial.params.keys()) if study.best_trial.params else []
            if params_to_slice:
                 fig_slice = optuna.visualization.plot_slice(study, params=params_to_slice)
                 # fig_slice.show()
                 fig_slice.write_html(os.path.join(plot_dir, "slice_plot.html"))
                 print(f"  Saved: {os.path.join(plot_dir, 'slice_plot.html')}")
            else:
                print("  Skipping slice plot: No parameters found in best_trial (or no best_trial).")
        except Exception as e:
            print(f"  Skipping slice plot: {e}")

    print("\nAnalysis complete. Check the console output and the 'optuna_plots' directory for HTML plots.")

if __name__ == "__main__":
    # Make sure these are installed in the environment where you run this script:
    # pip install optuna pandas plotly matplotlib
    # (matplotlib is sometimes a dependency of plotly's visualization, good to have)
    import os # for os.makedirs

    analyze_optuna_study(study_name=STUDY_NAME, storage_url=STORAGE_URL)