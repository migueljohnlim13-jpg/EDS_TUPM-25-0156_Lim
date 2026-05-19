import os
import sys
import warnings

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

warnings.filterwarnings("ignore")


# =============================================================================
# CLASS 1: DATA INGESTION
# =============================================================================
class DataIngestion:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.raw_df = None
        self.filtered_df = None

    def load(self) -> pd.DataFrame:
        try:
            self.raw_df = pd.read_csv(
                self.filepath,
                sep=';',
                decimal=','
            )

            print(
                f"[INGESTION] Dataset loaded successfully: "
                f"{self.raw_df.shape[0]} rows, "
                f"{self.raw_df.shape[1]} columns"
            )

            return self.raw_df

        except FileNotFoundError:
            print(f"[ERROR] CSV file not found: {self.filepath}")
            sys.exit(1)

        except Exception as e:
            print(f"[ERROR] Failed to load dataset: {e}")
            sys.exit(1)

    def apply_unique_filter(self) -> pd.DataFrame:
        if self.raw_df is None:
            raise ValueError("Dataset must be loaded before filtering.")

        self.filtered_df = self.raw_df.copy()
        return self.filtered_df


# =============================================================================
# CLASS 2: DATA CLEANING & FEATURE ENGINEERING
# =============================================================================
class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def remove_nulls(self):
        print("[CLEANING] Removing null values...")

        before = len(self.df)

        important_cols = [
            'Goal x',
            'Goal y',
            'Goal z',
            'measurement_x',
            'measurement_y',
            'measurement_z'
        ]

        existing_cols = [c for c in important_cols if c in self.df.columns]

        self.df.dropna(subset=existing_cols, how='all', inplace=True)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())

        print(f"[CLEANING] Removed {before - len(self.df)} invalid rows")

        return self

    def remove_duplicates(self):
        print("[CLEANING] Removing duplicate rows...")

        before = len(self.df)
        self.df.drop_duplicates(inplace=True)

        print(f"[CLEANING] Removed {before - len(self.df)} duplicates")

        return self

    def correct_dtypes(self):
        print("[CLEANING] Standardizing data types...")

        # Convert numeric columns safely
        for col in self.df.columns:
            if col != 'Timestamp':
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Handle timestamp
        if 'Timestamp' in self.df.columns:
            try:
                timestamps = (
                    self.df['Timestamp']
                    .astype(str)
                    .str.replace(',', '')
                    .str.replace('E+', '', regex=False)
                )

                self.df['Formatted_Timestamp'] = pd.to_datetime(
                    timestamps,
                    errors='coerce'
                )

            except Exception:
                self.df['Formatted_Timestamp'] = pd.date_range(
                    start='2020-01-01',
                    periods=len(self.df),
                    freq='s'
                )

        return self

    def engineer_features(self):
        print("[CLEANING] Engineering features...")

        required_cols = [
            'Goal x', 'Goal y', 'Goal z',
            'measurement_x', 'measurement_y', 'measurement_z'
        ]

        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Convert goal coordinates from meters to millimeters
        goal_x_mm = self.df['Goal x'] * 1000
        goal_y_mm = self.df['Goal y'] * 1000
        goal_z_mm = self.df['Goal z'] * 1000

        # Positional errors
        self.df['error_x'] = self.df['measurement_x'] - goal_x_mm
        self.df['error_y'] = self.df['measurement_y'] - goal_y_mm
        self.df['error_z'] = self.df['measurement_z'] - goal_z_mm

        # Euclidean distance error
        self.df['position_error'] = np.sqrt(
            self.df['error_x'] ** 2 +
            self.df['error_y'] ** 2 +
            self.df['error_z'] ** 2
        )

        # Orientation errors
        orientation_cols = [
            'Goal roll', 'Goal pitch', 'Goal yaw',
            'measurement_rx', 'measurement_ry', 'measurement_rz'
        ]

        if all(col in self.df.columns for col in orientation_cols):
            goal_roll_deg = np.degrees(self.df['Goal roll'])
            goal_pitch_deg = np.degrees(self.df['Goal pitch'])
            goal_yaw_deg = np.degrees(self.df['Goal yaw'])

            self.df['error_roll'] = self.df['measurement_rx'] - goal_roll_deg
            self.df['error_pitch'] = self.df['measurement_ry'] - goal_pitch_deg
            self.df['error_yaw'] = self.df['measurement_rz'] - goal_yaw_deg

            self.df['orientation_error'] = np.sqrt(
                self.df['error_roll'] ** 2 +
                self.df['error_pitch'] ** 2 +
                self.df['error_yaw'] ** 2
            )

        # Temperature deviation
        if 'measurement_temperature' in self.df.columns:
            self.df['temp_deviation'] = (
                self.df['measurement_temperature'] -
                self.df['measurement_temperature'].mean()
            )

        return self

    def save(self, out_dir='data'):
        print("[CLEANING] Saving cleaned dataset...")

        os.makedirs(out_dir, exist_ok=True)

        output_path = os.path.join(out_dir, 'cleaned_measurements.csv')

        self.df.to_csv(output_path, index=False)

        print(f"[CLEANING] Saved cleaned file to: {output_path}")

        return self

    def get_clean_df(self):
        return self.df


# =============================================================================
# CLASS 3: ANALYSIS
# =============================================================================
class Analyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def run_all(self):
        print("[ANALYSIS] Running statistical analysis...")

        results = {}

        numeric_df = self.df.select_dtypes(include=[np.number])

        results['correlation'] = numeric_df.corr()

        if (
            'measurement_temperature' in self.df.columns and
            'position_error' in self.df.columns
        ):
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    self.df['measurement_temperature'],
                    self.df['position_error']
                )

                results['temp_regression'] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'std_err': std_err
                }

                print(
                    f"[ANALYSIS] Regression completed | "
                    f"R² = {r_value ** 2:.4f}"
                )

            except Exception as e:
                print(f"[WARNING] Regression failed: {e}")

        results['summary'] = {
            'mean_position_error': self.df['position_error'].mean(),
            'max_position_error': self.df['position_error'].max(),
            'std_position_error': self.df['position_error'].std()
        }

        return results


# =============================================================================
# CLASS 4: VISUALIZATION
# =============================================================================
class Visualizer:
    def __init__(self, df: pd.DataFrame, out_dir='outputs'):
        self.df = df
        self.out_dir = out_dir

        os.makedirs(self.out_dir, exist_ok=True)

    def generate_static_plots(self, corr_matrix):
        print("[VISUALIZATION] Creating static plots...")

        # Heatmap
        plt.figure(figsize=(10, 8))

        plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar()

        plt.xticks(
            range(len(corr_matrix.columns)),
            corr_matrix.columns,
            rotation=90,
            fontsize=6
        )

        plt.yticks(
            range(len(corr_matrix.columns)),
            corr_matrix.columns,
            fontsize=6
        )

        plt.title('Correlation Matrix')
        plt.tight_layout()

        heatmap_path = os.path.join(self.out_dir, 'correlation_heatmap.png')
        plt.savefig(heatmap_path, dpi=150)
        plt.close()

        print(f"[VISUALIZATION] Heatmap saved: {heatmap_path}")

        # Scatter Plot
        if (
            'measurement_temperature' in self.df.columns and
            'position_error' in self.df.columns
        ):
            plt.figure(figsize=(8, 6))

            plt.scatter(
                self.df['measurement_temperature'],
                self.df['position_error'],
                alpha=0.6
            )

            plt.xlabel('Temperature')
            plt.ylabel('Position Error')
            plt.title('Temperature vs Position Error')
            plt.grid(True)
            plt.tight_layout()

            scatter_path = os.path.join(
                self.out_dir,
                'temperature_vs_error.png'
            )

            plt.savefig(scatter_path, dpi=150)
            plt.close()

            print(f"[VISUALIZATION] Scatter plot saved: {scatter_path}")

    def generate_kinematic_animation(self):
        print("[VISUALIZATION] Generating animation...")

        try:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            anim_data = self.df.head(100)

            target_x = anim_data['Goal x'] * 1000
            target_y = anim_data['Goal y'] * 1000
            target_z = anim_data['Goal z'] * 1000

            meas_x = anim_data['measurement_x']
            meas_y = anim_data['measurement_y']
            meas_z = anim_data['measurement_z']

            line1, = ax.plot([], [], [], 'g--', label='Target')
            line2, = ax.plot([], [], [], 'b-', label='Measured')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()

            def update(frame):
                line1.set_data(target_x[:frame], target_y[:frame])
                line1.set_3d_properties(target_z[:frame])

                line2.set_data(meas_x[:frame], meas_y[:frame])
                line2.set_3d_properties(meas_z[:frame])

                return line1, line2

            ani = animation.FuncAnimation(
                fig,
                update,
                frames=len(anim_data),
                interval=100,
                blit=False
            )

            gif_path = os.path.join(self.out_dir, 'trajectory.gif')

            ani.save(gif_path, writer='pillow', fps=10)

            plt.close()

            print(f"[VISUALIZATION] Animation saved: {gif_path}")

        except Exception as e:
            print(f"[WARNING] Animation failed: {e}")

    def generate_cv2_frame_analysis(self):
        print("[VISUALIZATION] Generating OpenCV video...")

        try:
            width, height = 640, 480

            video_path = os.path.join(self.out_dir, 'frame_analysis.avi')

            fourcc = cv2.VideoWriter_fourcc(*'XVID')

            video = cv2.VideoWriter(
                video_path,
                fourcc,
                2.0,
                (width, height)
            )

            samples = self.df.head(20)

            for idx, row in samples.iterrows():
                frame = np.ones((height, width, 3), dtype=np.uint8) * 255

                cv2.putText(
                    frame,
                    f'Frame: {idx + 1}',
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2
                )

                if 'position_error' in row:
                    cv2.putText(
                        frame,
                        f'Position Error: {row["position_error"]:.3f}',
                        (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                video.write(frame)

            video.release()

            print(f"[VISUALIZATION] Video saved: {video_path}")

        except Exception as e:
            print(f"[WARNING] OpenCV rendering failed: {e}")


# =============================================================================
# MAIN PROGRAM
# =============================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('ROBOTIC POSE TRACKING ANALYZER')
    print('=' * 60)

    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    # Safe absolute CSV path
    base_dir = os.path.dirname(os.path.abspath(__file__))

    csv_filename = os.path.join(
        base_dir,
        'data\measurements_joint_configurations.csv'
    )

    # Verify file exists
    if not os.path.exists(csv_filename):
        print(f'[CRITICAL ERROR] CSV not found: {csv_filename}')
        sys.exit(1)

    # STEP 1 — INGESTION
    print('\n[STEP 1] DATA INGESTION')

    ingestion = DataIngestion(csv_filename)

    raw_df = ingestion.load()
    filtered_df = ingestion.apply_unique_filter()

    # STEP 2 — CLEANING
    print('\n[STEP 2] DATA CLEANING')

    cleaner = DataCleaner(filtered_df)

    cleaner \
        .remove_nulls() \
        .remove_duplicates() \
        .correct_dtypes() \
        .engineer_features() \
        .save()

    clean_df = cleaner.get_clean_df()

    # STEP 3 — ANALYSIS
    print('\n[STEP 3] ANALYSIS')

    analyzer = Analyzer(clean_df)

    results = analyzer.run_all()

    # STEP 4 — VISUALIZATION
    print('\n[STEP 4] VISUALIZATION')

    visualizer = Visualizer(clean_df)

    visualizer.generate_static_plots(results['correlation'])
    visualizer.generate_kinematic_animation()
    visualizer.generate_cv2_frame_analysis()

    print('\n' + '=' * 60)
    print('PIPELINE EXECUTED SUCCESSFULLY')
    print('Outputs saved to:')
    print('- data/')
    print('- outputs/')
    print('=' * 60)
