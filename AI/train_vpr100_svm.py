try:
    from AI.training_presets import VPR100_PRESET, run_svm_cli
except ModuleNotFoundError:
    from training_presets import VPR100_PRESET, run_svm_cli  # type: ignore


if __name__ == "__main__":
    run_svm_cli(VPR100_PRESET)
