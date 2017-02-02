import numpy as np
from pytest import mark
from scipy.stats import multivariate_normal, norm

from src.ripple_decoding import (_fix_zero_bins, evaluate_mark_space,
                                 _normalize_column_probability,
                                 combined_likelihood,
                                 normalize_to_probability,
                                 estimate_place_field,
                                 estimate_ground_process_intensity,
                                 estimate_place_occupancy,
                                 poisson_mark_likelihood)


def test_evaluate_mark_space():
    n_marks, n_training_spikes, mark_std_deviation = 4, 10, 1

    test_marks = np.arange(1, 9, 2)

    training_marks = np.zeros((n_training_spikes, n_marks))
    training_marks[3, :] = np.arange(9, 17, 2)

    mark_space_estimator = evaluate_mark_space(
        test_marks, training_marks=training_marks,
        mark_std_deviation=mark_std_deviation)

    expected_mark1 = multivariate_normal(
        mean=np.arange(9, 17, 2),
        cov=np.identity(n_marks) * mark_std_deviation).pdf(
            np.arange(1, 9, 2))
    expected_mark2 = multivariate_normal(
        mean=np.zeros(n_marks,),
        cov=np.identity(n_marks) * mark_std_deviation).pdf(
            np.arange(1, 9, 2))
    expected = np.ones((n_training_spikes,)) * expected_mark2
    expected[3] = expected_mark1

    assert np.allclose(mark_space_estimator, expected)


def test__normalize_column_probability():
    '''All columns should sum to one after normalization
    '''
    transition_matrix = np.arange(1, 10).reshape(3, 3)
    expected = np.ones((3,))
    assert np.allclose(
        _normalize_column_probability(transition_matrix).sum(axis=0),
        expected)


def test__fix_zero_bins():
    '''A column of all zeros should be set to ones
    '''
    transition_matrix = np.arange(0, 9).reshape(3, 3)
    transition_matrix[:, 2] = 0
    expected = np.copy(transition_matrix)
    expected[:, 2] = 1
    assert np.allclose(_fix_zero_bins(transition_matrix), expected)


def test_normalize_to_probability():
    ''' A vector should normalize to one
    '''
    x = np.arange(1, 9)
    assert normalize_to_probability(x).sum() == 1


@mark.parametrize('data, exponent, expected', [
    (np.arange(1, 9), 1, np.nanprod(np.arange(1, 9))),
    (np.arange(1, 9), 2, np.nanprod(np.arange(1, 9) ** 2)),  # test kwarg
    (np.arange(1, 9).reshape(2, 4), 2,  # test product along 1st dimension
     np.nanprod(np.arange(1, 9).reshape(2, 4) ** 2, axis=0)),
    (2, 2, 4),  # test single data point
])
def test_combined_likelihood(data, exponent, expected):
    def likelihood_function(x, exponent=1):
        return x ** exponent
    assert np.allclose(
        combined_likelihood(data, likelihood_function,
                            likelihood_kwargs=dict(exponent=exponent)),
        expected)


def test_get_ground_process_intensity():
    place_field_estimator = np.zeros((61, 2))
    place_field_estimator[30, 0] = 1
    place_field_estimator[31, 1] = 1

    place_occupancy = np.ones((61,))
    place_occupancy[31] = 4
    ground_process_intensity = estimate_ground_process_intensity(
        place_field_estimator, place_occupancy)

    expected = np.zeros((61,))
    expected[30] = 0.8
    expected[31] = 0.2

    assert np.allclose(ground_process_intensity, expected)


def test_estimate_place_field():
    place_bins = np.linspace(0, 150, 61)
    place_at_spike = np.asarray([25, 100])
    place_std_deviation = 20
    place_field_estimator = estimate_place_field(
        place_bins, place_at_spike,
        place_std_deviation=place_std_deviation)

    expected1 = norm.pdf(
        place_bins, place_at_spike[0], place_std_deviation)
    expected2 = norm.pdf(
        place_bins, place_at_spike[1], place_std_deviation)

    assert np.allclose(place_field_estimator,
                       np.vstack((expected1, expected2)).T)


def test_estimate_place_occupancy():
    place_bins = np.linspace(0, 150, 61)
    place = np.asarray([25, 100])
    place_std_deviation = 20
    place_occupancy = estimate_place_occupancy(
        place_bins, place, place_std_deviation=place_std_deviation)
    expected1 = norm.pdf(
        place_bins, place[0], place_std_deviation)
    expected2 = norm.pdf(
        place_bins, place[1], place_std_deviation)
    assert np.allclose(place_occupancy, expected1 + expected2)


def test_poisson_mark_likelihood_is_spike():
    def identity(marks):
        return marks

    n_signals, n_marks, n_parameters = 10, 4, 4

    marks = (np.ones((n_signals, n_marks)) *
             np.arange(0, n_signals)[:, np.newaxis])
    no_spike_ind = 5
    marks[no_spike_ind, :] = np.nan

    ground_process_intensity = np.zeros((n_signals, n_parameters))

    likelihood = poisson_mark_likelihood(
        marks, joint_mark_intensity=identity,
        ground_process_intensity=ground_process_intensity)

    expected_likelihood = np.copy(marks)
    expected_likelihood[no_spike_ind, :] = 1

    assert np.allclose(likelihood, expected_likelihood)
    assert np.allclose(likelihood, expected_likelihood)
