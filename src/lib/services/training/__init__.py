"""
lib.services.training — CNN training, dataset generation, and simulation modules.

This package contains GPU-heavy training code that runs on a dedicated
training machine. All imports are lazy/guarded
so that importing ``lib.services.training`` on a CPU-only deployment doesn't pull
in torch or other heavy dependencies.

Sub-modules:
    dataset_generator  — Orchestrates chart rendering + auto-labeling
    rb_simulator      — Replays historical bars through bracket logic
    trainer_server     — FastAPI server that accepts POST /train requests

Usage::

    from lib.services.training.rb_simulator import (
        simulate_batch,
        simulate_batch_prev_day,
        simulate_batch_ib,
        simulate_batch_consolidation,
        BracketConfig,
        RBSimResult,
        ORBSimResult,  # backward-compat alias
    )

    from lib.services.training.dataset_generator import (
        generate_dataset,
        DatasetConfig,
        DatasetStats,
    )
"""
