import adore_doe


def test_import_and_version():
    assert isinstance(adore_doe.__version__, str) and adore_doe.__version__
    for name in ("core", "design", "screening", "active_learning"):
        assert hasattr(adore_doe, name)


def test_design_public_api_is_exported():
    import adore_doe.design as D
    for name in ("ccd", "i_optimal", "d_optimal", "augment", "generate_candidates", "Region",
                 "IntegrationDomain", "ModelSpec", "fds", "plot_fds", "check_design", "i_criterion"):
        assert hasattr(D, name), name
    import adore_doe.active_learning as AL
    for name in ("GaussianProcess", "ActiveLearner", "MultiResponseStopping", "select_batch", "EvaluationSet",
                 "gpr_cv_r2", "predictive_r2", "STRATEGIES"):
        assert hasattr(AL, name), name
    from adore_doe.core import SeedManager, derive_seed, resolve_seed  # noqa: F401
    assert hasattr(adore_doe, "InfeasibleRegionError")


def test_core_import_does_not_need_matplotlib():
    """ADR-008: matplotlib is optional; importing the package must work without it and
    plotting must fail with an actionable message."""
    import os
    import subprocess
    import sys
    src = os.path.dirname(os.path.dirname(adore_doe.__file__))
    code = ("import sys; sys.modules['matplotlib'] = None; sys.modules['matplotlib.pyplot'] = None\n"
            "import adore_doe, adore_doe.screening, adore_doe.design, adore_doe.active_learning\n"
            "from adore_doe.design import ccd, fds\n"
            "try:\n    from adore_doe.design import plots\n"
            "except ImportError as e:\n    print('MSG', e)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": src})
    assert out.returncode == 0, out.stderr
    assert "adore-doe[plot]" in out.stdout
