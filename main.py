import argparse
import config
from utils import save_to_json, create_run_dir, setup_logger
from data import aux_download_datasets
from phases import run_phase_1, run_phase_2, run_phase_3, run_phase_4, run_phase_5

def main():
    """
    Main entry point for the Lattice-Preserving QCNN experimental suite.
    
    This script orchestrates the execution of five distinct experimental phases:
    1.  Phase 1 (Filter Search): Determines the optimal number of quantum filters.
    2.  Phase 2 (Dynamics Search): Finds the optimal time evolution parameters using Phase 1 results.
    3.  Phase 3 (Deep Analysis): Trains the best models fully (20 epochs) and computes physical metrics (BCH, Fisher).
    4.  Phase 4 (Data Efficiency): Evaluates performance vs. training set size using optimal configs.
    5.  Phase 5 (Binary Stability): Tests model consistency on binary classification tasks.
    """
    
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Run LP-QCNN Experiments")
    parser.add_argument('--mode', type=str, default='DUMMY', choices=['DUMMY', 'FULL'],
                        help="Execution mode: 'DUMMY' for quick testing, 'FULL' for complete experiments.")
    args = parser.parse_args()
    
    # Update Global Configuration
    config.MODE = args.mode
    
    # 2. Setup Environment (Directories & Logging)
    create_run_dir()
    setup_logger()
    
    config.LOGGER.info(f"\n" + "="*60)
    config.LOGGER.info(f"[START] Running Experiment Suite in >>> {config.MODE} <<< mode")
    config.LOGGER.info(f"[INFO] All outputs will be saved to: {config.RUN_DIR}")
    config.LOGGER.info("="*60 + "\n")
    
    # 3. Data Preparation
    aux_download_datasets()
    
    # 4. Experimental Phases Execution
    
    # --- Phase 1: Optimal Filter Search ---
    # Input: None
    # Returns: dict {'MNIST': optimal_filter_int, ...}
    p1_best_filters = run_phase_1()
    
    # --- Phase 2: Dynamics Grid Search ---
    # Input: Best filters from Phase 1
    # Returns: dict {'MNIST': full_config_dict, ...} containing optimal Time, Steps, and Filter
    p2_best_configs = run_phase_2(p1_best_filters)
    
    # --- Phase 3: Deep Analysis ---
    # Input: Best full configs from Phase 2
    # Action: Extended training (20 epochs), Physics Analysis (BCH, Fisher), Plotting
    run_phase_3(p2_best_configs)
    
    # --- Phase 4: Data Efficiency ---
    # Input: Best full configs from Phase 2
    # Action: Train on [1000, 2000, ..., Full] samples to test sample complexity
    run_phase_4(p2_best_configs)
    
    # --- Phase 5: Binary Stability ---
    # Input: None (Uses independent configuration)
    # Action: repeated runs on binary classes (0 vs 1) to test variance
    run_phase_5() 
    
    # 5. Finalize
    save_to_json(config.GLOBAL_RESULTS)
    config.LOGGER.info(f"\n[FINISH] All experiments completed. Results saved in {config.RUN_DIR}")

if __name__ == "__main__":
    main()