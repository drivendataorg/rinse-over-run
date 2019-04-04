import pandas as pd

phases = pd.DataFrame(
    {
        "phase": ["pre_rinse", "caustic", "intermediate_rinse", "acid", "final_rinse"],
        "phase_id": range(1, 6),
    }
)
