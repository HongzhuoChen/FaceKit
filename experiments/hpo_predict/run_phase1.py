"""Phase-1 orchestrator: runs steps 1-7 in one seeded, timestamped run.

Step 1-2 write committed label artifacts under labels/. Steps 3-7 write seeded run
artifacts under results/<exp_name>/<YYYY-MM-DD_HH-MM-SS>/. Each step prints its
observable verify block.
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
import step1_materialize_gold  # noqa: E402
import step2_build_labels  # noqa: E402
import step3_split  # noqa: E402
import step4_rule_baseline  # noqa: E402
import step5_train_model  # noqa: E402
import step6_evaluate  # noqa: E402
import step7_compare  # noqa: E402


def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = config.RESULTS_ROOT / config.EXP_NAME / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Phase-1 run: {run_dir} (seed={config.SEED}) ===")

    step1_materialize_gold.run()
    step2_build_labels.run()
    step3_split.run(run_dir)
    step4_rule_baseline.run(run_dir)
    step5_train_model.run(run_dir)
    step6_evaluate.run(run_dir)
    step7_compare.run(run_dir)
    print(f"=== done -> {run_dir} ===")


if __name__ == "__main__":
    main()
