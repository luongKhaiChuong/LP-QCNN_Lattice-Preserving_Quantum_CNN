import argparse
import config
from utils import save_to_json, create_run_dir, setup_logger
from data import aux_download_datasets
from phases import run_phase_1, run_phase_2, run_phase_3, run_phase_4

def main():
    """
    Main entry point for the Lattice-Preserving QCNN experimental suite.
    
    This script orchestrates the execution of three distinct experimental phases:
    1.  Phase 1 (Full train): Train the models fully (20 epochs) and computes physical metrics (BCH, Fisher).
    2.  Phase 2 (Data Efficiency): Evaluates performance vs. training set size using optimal configs.
    3.  Phase 3 (Binary Stability): Tests model consistency on binary classification tasks.
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
    
    # --- Phase 1: Deep Analysis ---
    # Input: Best full configs from trials
    # Action: Extended training (20 epochs), Physics Analysis (BCH, Fisher), Plotting
    best_configs = {
        'MNIST': {'dataset': 'MNIST',
                  'time': 2.0, 
                  'time_steps': 4, 
                  'n_filters': 3}, 
        'FashionMNIST': {'dataset': 'FashionMNIST', 
                         'time': 5.0, 
                         'time_steps': 5, 
                         'n_filters': 6}
    }
    # run_phase_1(best_configs)
    
    # # --- Phase 2: Data Efficiency ---
    # # Input: Best full configs from Phase 2
    # # Action: Train on [1000, 2000, ..., Full] samples to test sample complexity
    # run_phase_2(best_configs)
    
    # # --- Phase 3: Binary Stability ---
    # # Input: None (Uses independent configuration)
    # # Action: repeated runs on binary classes (0 vs 1) to test variance
    # run_phase_3() 
    
    run_phase_4(best_configs)

        
    # Finalize
    save_to_json(config.GLOBAL_RESULTS)
    config.LOGGER.info(f"\n[FINISH] All experiments completed. Results saved in {config.RUN_DIR}")

if __name__ == "__main__":
    main()