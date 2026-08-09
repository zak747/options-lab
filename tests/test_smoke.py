"""Smoke test. Proves the package installs and imports before any real
work begins — debugging a broken import at the same time as a first
pricer is miserable."""


def test_package_imports():
    import optionslab

    assert optionslab.__version__


def test_all_modules_import():
    from optionslab import bs, chain, config, hedge, iv, lsm, mc, surface, vix

    assert config.ROOT.exists()
