from fairqmodel.extract_predict_load import get_batches_to_reschedule


def test_get_batches_to_reschedule():
    reporting = (
        {"batch": 1, "finished": False},
        {"batch": 2, "finished": True},
        {"batch": 3, "finished": False},
        {"batch": 1, "finished": True},
    )

    res = get_batches_to_reschedule(reporting)

    assert res == [3]
