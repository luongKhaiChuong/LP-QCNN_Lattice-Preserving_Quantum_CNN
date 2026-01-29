import argparse
import config
from utils import save_to_json, create_run_dir, setup_logger
from data import aux_download_data
from phases import run_phase_1, run_phase_2, run_phase_3, run_phase_4, run_phase_5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='DUMMY', choices=['DUMMY', 'FULL'])
    args = parser.parse_args()
    config.MODE = args.mode
    
    create_run_dir()
    setup_logger()
    
    config.LOGGER.info(f"\n" + "="*60)
    config.LOGGER.info(f"[START] Running in >>> {config.MODE} <<< mode")
    config.LOGGER.info(f"[INFO] Saving all results to: {config.RUN_DIR}")
    config.LOGGER.info("="*60 + "\n")
    
    aux_download_data()
    
    p1_best = run_phase_1()
    p2_best = run_phase_2(p1_best)
    run_phase_3(p1_best)
    run_phase_4(p1_best)
    run_phase_5() 
    
    save_to_json(config.GLOBAL_RESULTS)
    config.LOGGER.info(f"\n[FINISH] Results saved in {config.RUN_DIR}")