try:
    from AI.training_presets import C4001_PRESET, run_svm_cli
except ModuleNotFoundError:
    from training_presets import C4001_PRESET, run_svm_cli  # type: ignore


if __name__ == "__main__":
    run_svm_cli(C4001_PRESET)
