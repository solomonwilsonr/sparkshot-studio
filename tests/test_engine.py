from app.engine import Shot, PRESETS

def test_presets():
    assert 'push_in' in PRESETS
    assert 'dollies forward' in Shot('A cat', 'push_in').expanded_prompt()
