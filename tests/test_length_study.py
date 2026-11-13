"""Guard against leakage and noncausal previous-turn features."""
from scripts.study_lengths import evaluate, predictions, split_sessions


def test_split_is_session_disjoint_and_reproducible():
    sessions = {f's{i}' for i in range(20)}
    train, test = split_sessions(sessions, 11)
    assert train.isdisjoint(test)
    assert train | test == sessions
    assert (train, test) == split_sessions(reversed(sorted(sessions)), 11)


def test_previous_turn_is_causal_and_session_local():
    rows = [dict(session_id=s, prompt='hello', output_length=y)
            for s, y in [('a', 10), ('b', 999), ('a', 30), ('b', 5)]]
    preds = [p['previous_turn'] for _, p in predictions(rows, 50)]
    assert preds == [50, 50, 10, 999]


def test_test_labels_do_not_change_training_thresholds_or_predictions():
    rows = [dict(session_id=s, prompt='hello', model='test', output_length=y)
            for s, y in [('train', 10), ('train', 30), ('test', 100)]]
    first, raw = evaluate(rows, {'train'}, {'test'})
    rows[-1]['output_length'] = 10000
    second, changed = evaluate(rows, {'train'}, {'test'})
    assert first['training_median'] == second['training_median'] == 20
    assert first['training_p90'] == second['training_p90']
    assert raw[0]['predictions'] == changed[0]['predictions']
