"""Builds one index generation: `python -m data_layer.vector_db_manager.index_build SPEC`."""

import json
import os
import sys


def _yield_to_the_application() -> None:
    # If memory runs out despite the estimates, the kernel's OOM killer takes
    # the process with the highest score: this one, not the application.
    try:
        with open("/proc/self/oom_score_adj", "w") as adjust:
            adjust.write("1000")
    except OSError:
        pass
    try:
        os.nice(10)
    except OSError:
        pass


def build(spec: dict) -> None:
    import diskannpy
    import numpy as np

    if spec["kind"] == "memory":
        diskannpy.build_memory_index(
            spec["vectors"], spec["metric"], spec["directory"], spec["complexity"],
            spec["degree"], spec["threads"], vector_dtype=np.float32, index_prefix="ann",
        )
    else:
        diskannpy.build_disk_index(
            spec["vectors"], spec["metric"], spec["directory"], spec["complexity"],
            spec["degree"], spec["pq_gb"], spec["build_gb"], spec["threads"],
            vector_dtype=np.float32, index_prefix="ann",
        )


if __name__ == "__main__":
    _yield_to_the_application()
    build(json.loads(sys.argv[1]))
