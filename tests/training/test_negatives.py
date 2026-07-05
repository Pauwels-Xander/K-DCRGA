import torch

from kdcrga.training.negatives import sample_negatives


def test_sample_negatives_shapes_and_filtering():
    pos_pair = torch.tensor([[0, 1, 2], [1, 2, 3]])
    pos_se = torch.tensor([0, 0, 1])
    positives = {(0, 1, 0), (1, 2, 0), (2, 3, 1)}
    neg_pair, neg_se = sample_negatives(
        pos_pair, pos_se, num_drugs=4, positives=positives, seed=0
    )
    assert neg_pair.shape == pos_pair.shape
    assert neg_se.shape == pos_se.shape
    assert torch.equal(neg_se, pos_se)
    for i in range(neg_pair.size(1)):
        a, b, k = int(neg_pair[0, i]), int(neg_pair[1, i]), int(neg_se[i])
        assert (a, b, k) not in positives
        assert a < 4 and b < 4


def test_sample_negatives_no_self_loops():
    """Every successfully-corrupted negative must have neg_a != neg_b."""
    pos_pair = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    pos_se = torch.tensor([0, 0, 1, 1])
    positives = {(0, 1, 0), (1, 2, 0), (2, 3, 1), (3, 0, 1)}
    neg_pair, _ = sample_negatives(
        pos_pair, pos_se, num_drugs=10, positives=positives, seed=42
    )
    for i in range(neg_pair.size(1)):
        a, b = int(neg_pair[0, i]), int(neg_pair[1, i])
        assert a != b, f"Self-loop found at index {i}: ({a}, {b})"


def test_sample_negatives_determinism():
    """Same seed -> identical output; different seed -> different output.

    The different-seed check uses num_drugs=200 and e=10 positives so the
    probability of a spurious collision is negligible (< 1e-20).
    """
    g = torch.Generator().manual_seed(42)
    pos_pair = torch.randint(0, 200, (2, 10), generator=g)
    pos_se = torch.randint(0, 10, (10,), generator=g)
    positives: set = set()
    neg1a, _ = sample_negatives(pos_pair, pos_se, num_drugs=200, positives=positives, seed=7)
    neg1b, _ = sample_negatives(pos_pair, pos_se, num_drugs=200, positives=positives, seed=7)
    assert torch.equal(neg1a, neg1b), "Same seed must yield identical output"

    neg2, _ = sample_negatives(pos_pair, pos_se, num_drugs=200, positives=positives, seed=99)
    # With 200 drugs and 10 pairs, a spurious all-element collision is effectively impossible
    assert not torch.equal(neg1a, neg2), "Different seeds should yield different output"


def test_sample_negatives_side_effects_unchanged():
    """neg_side_effect must equal pos_side_effect (a clone, not the same object)."""
    pos_pair = torch.tensor([[0, 1], [2, 3]])
    pos_se = torch.tensor([5, 7])
    positives: set = set()
    neg_pair, neg_se = sample_negatives(
        pos_pair, pos_se, num_drugs=10, positives=positives, seed=0
    )
    assert torch.equal(neg_se, pos_se)
    # Must be a distinct tensor object (clone), not the same storage
    assert neg_se.data_ptr() != pos_se.data_ptr()


def test_sample_negatives_empty_input():
    """Empty input (zero edges) should return zero-sized tensors without error."""
    pos_pair = torch.zeros((2, 0), dtype=torch.long)
    pos_se = torch.zeros(0, dtype=torch.long)
    positives: set = set()
    neg_pair, neg_se = sample_negatives(
        pos_pair, pos_se, num_drugs=10, positives=positives, seed=0
    )
    assert neg_pair.shape == (2, 0)
    assert neg_se.shape == (0,)


def test_sample_negatives_output_shape_property():
    """Output shapes must always be (2, E) and (E,) for any non-empty input."""
    g = torch.Generator().manual_seed(0)
    pos_pair = torch.randint(0, 50, (2, 17), generator=g)
    pos_se = torch.randint(0, 5, (17,), generator=g)
    positives: set = set()
    neg_pair, neg_se = sample_negatives(
        pos_pair, pos_se, num_drugs=50, positives=positives, seed=13
    )
    assert neg_pair.shape == (2, 17), f"Expected (2, 17), got {neg_pair.shape}"
    assert neg_se.shape == (17,), f"Expected (17,), got {neg_se.shape}"


def test_sample_negatives_exhaustion_fallback():
    """When every candidate is rejected the sampler falls back to the positive's
    own endpoints rather than raising.

    Setup: num_drugs=2, single positive (0, 1, 0).  Corrupting drug 0 → drug 1
    gives (1, 1, 0) which is a self-loop (rejected), and corrupting drug 1 →
    drug 0 gives (0, 0, 0) which is also a self-loop.  There is no valid
    negative, so max_tries is exhausted and the fallback triggers.
    """
    pos_pair = torch.tensor([[0], [1]], dtype=torch.long)
    pos_se = torch.tensor([0], dtype=torch.long)
    positives: set[tuple[int, int, int]] = {(0, 1, 0)}
    neg_pair, neg_se = sample_negatives(
        pos_pair, pos_se, num_drugs=2, positives=positives, seed=0, max_tries=5
    )
    # Shapes must be correct
    assert neg_pair.shape == (2, 1), f"Expected (2, 1), got {neg_pair.shape}"
    assert neg_se.shape == (1,), f"Expected (1,), got {neg_se.shape}"
    # Fallback: negative equals the original positive pair
    assert int(neg_pair[0, 0]) == 0 and int(neg_pair[1, 0]) == 1, (
        f"Expected fallback (0,1), got ({int(neg_pair[0,0])},{int(neg_pair[1,0])})"
    )


def test_sample_negatives_rejection_under_dense_positives():
    """With a dense positives set the sampler must reject several candidates, but
    every returned negative must either be absent from ``positives`` or equal the
    original positive (fallback), and non-fallback entries must have neg_a != neg_b.
    """
    num_drugs = 5
    # Build a dense positives set: all ordered pairs for side-effect 0
    positives: set[tuple[int, int, int]] = {
        (a, b, 0) for a in range(num_drugs) for b in range(num_drugs) if a != b
    }
    # Leave side-effect 1 completely open so non-fallback negatives can exist
    pos_pair = torch.tensor(
        [[0, 1, 2, 3, 0], [1, 2, 3, 4, 4]], dtype=torch.long
    )
    pos_se = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    neg_pair, neg_se = sample_negatives(
        pos_pair, pos_se, num_drugs=num_drugs, positives=positives, seed=0, max_tries=20
    )
    assert neg_pair.shape == pos_pair.shape
    assert neg_se.shape == pos_se.shape
    for i in range(neg_pair.size(1)):
        a_neg = int(neg_pair[0, i])
        b_neg = int(neg_pair[1, i])
        k = int(neg_se[i])
        a_pos = int(pos_pair[0, i])
        b_pos = int(pos_pair[1, i])
        is_fallback = (a_neg == a_pos and b_neg == b_pos)
        if not is_fallback:
            assert (a_neg, b_neg, k) not in positives, (
                f"Entry {i}: ({a_neg},{b_neg},{k}) is in positives but is not a fallback"
            )
            assert a_neg != b_neg, (
                f"Entry {i}: self-loop ({a_neg},{b_neg}) in non-fallback negative"
            )
