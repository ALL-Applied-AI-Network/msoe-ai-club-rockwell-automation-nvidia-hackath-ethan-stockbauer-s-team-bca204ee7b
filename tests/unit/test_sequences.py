import pandas as pd

from mixing_copilot.features.sequences import build_causal_sequences


def test_causal_sequences_never_include_future_rows():
    frame = pd.DataFrame(
        {
            "vial_id": [1, 1, 1],
            "corrected_media_seconds": [1.0, 2.0, 3.0],
            "feature": [10.0, 20.0, 30.0],
        }
    )
    result = build_causal_sequences(frame, ["feature"], history=3)
    assert result.values[0, :, 0].tolist() == [0.0, 0.0, 10.0]
    assert result.values[1, :, 0].tolist() == [0.0, 10.0, 20.0]
    assert result.values[2, :, 0].tolist() == [10.0, 20.0, 30.0]
