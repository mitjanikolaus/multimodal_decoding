import argparse
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from nilearn import datasets
from nilearn.decoding.searchlight import GroupIterator
from nilearn.surface import surface

from sklearn import neighbors
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer
import os
import pickle

from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyses.ridge_regression_decoding import TRAIN_MODE_CHOICES, FEATS_SELECT_DEFAULT, \
    FEATURE_COMBINATION_CHOICES, VISION_FEAT_COMBINATION_CHOICES, DEFAULT_SUBJECTS, get_nn_latent_data, \
    get_default_features, pairwise_accuracy

from utils import VISION_MEAN_FEAT_KEY, IDS_IMAGES_TEST, SURFACE_LEVEL_FMRI_DIR

DEFAULT_N_JOBS = 10

NUM_TEST_STIMULI = 140
INDICES_TEST_STIM_CAPTION = list(range(NUM_TEST_STIMULI // 2))
INDICES_TEST_STIM_IMAGE = list(range(NUM_TEST_STIMULI // 2, NUM_TEST_STIMULI))
IDS_TEST_STIM = np.array(IDS_IMAGES_TEST + IDS_IMAGES_TEST)

SEARCHLIGHT_OUT_DIR = os.path.expanduser("~/data/multimodal_decoding/searchlight/")


def custom_cross_val_score(
        estimator,
        X,
        y=None,
        *,
        groups=None,
        scoring=None,
        cv=None,
        n_jobs=None,
        verbose=0,
        fit_params=None,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
):
    """Evaluate a score by cross-validation.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only
        a single value.

        Similar to :func:`cross_validate`
        but only a single metric is permitted.

        If `None`, the estimator's default scorer (if available) is used.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - `None`, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable that generates (train, test) splits as arrays of indices.

        For `int`/`None` inputs, if the estimator is a classifier and `y` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            `cv` default value if `None` changed from 3-fold to 5-fold.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        The verbosity level.

    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - ``None``, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

        .. versionadded:: 0.20

    Returns
    -------
    scores : ndarray of float of shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.

    See Also
    --------
    cross_validate : To run cross-validation on multiple metrics and also to
        return train scores, fit times and score times.

    cross_val_predict : Get predictions from each split of cross-validation for
        diagnostic purposes.

    sklearn.metrics.make_scorer : Make a scorer from a performance metric or
        loss function.

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_val_score
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()
    >>> print(cross_val_score(lasso, X, y, cv=3))
    [0.3315057  0.08022103 0.03531816]
    """
    # To ensure multimetric format is not supported
    # scorer = check_scoring(estimator, scoring=scoring)

    cv_results = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        groups=groups,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        fit_params=fit_params,
        pre_dispatch=pre_dispatch,
        error_score=error_score,
    )
    return cv_results


def custom_group_iter_search_light(
        list_rows,
        estimator,
        X,
        y,
        groups,
        scoring,
        cv,
        thread_id,
        total,
        print_interval=500,
):
    """Perform grouped iterations of search_light.

    Parameters
    ----------
    list_rows : array of arrays of int
        adjacency rows. For a voxel with index i in X, list_rows[i] is the list
        of neighboring voxels indices (in X).

    estimator : estimator object implementing 'fit'
        object to use to fit the data

    X : array-like of shape at least 2D
        data to fit.

    y : array-like
        target variable to predict.

    groups : array-like, optional
        group label for each sample for cross validation.

    scoring : string or callable, optional
        Scoring strategy to use. See the scikit-learn documentation.
        If callable, takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross validation is
        used or 3-fold stratified cross-validation when y is supplied.

    thread_id : int
        process id, used for display.

    total : int
        Total number of voxels, used for display

    print_interval : int, default=500
        The interval for printing progress information.

    Returns
    -------
    par_scores : numpy.ndarray
        score for each voxel. dtype: float64.
    """
    par_scores = []
    t0 = time.time()
    for i, row in enumerate(list_rows):
        kwargs = {"scoring": scoring, "groups": groups}
        scores = custom_cross_val_score(estimator, X[:, row], y, cv=cv, n_jobs=1, verbose=0, **kwargs)
        par_scores.append({key: np.mean(score) for key, score in scores.items()})
        if print_interval > 0:
            if i % print_interval == 0:
                # If there is only one job, progress information is fixed
                crlf = "\r" if total == len(list_rows) else "\n"
                percent = float(i) / len(list_rows)
                percent = round(percent * 100, 2)
                dt = time.time() - t0
                # We use a max to avoid a division by zero
                remaining = (100.0 - percent) / max(0.01, percent) * dt
                sys.stderr.write(
                    f"Job #{thread_id}, processed {i}/{len(list_rows)} voxels "
                    f"({percent:0.2f}%, {round(remaining/60)} minutes remaining){crlf}"
                )
    return par_scores


def custom_search_light(
        X,
        y,
        estimator,
        A,
        groups=None,
        scoring=None,
        cv=None,
        n_jobs=-1,
        verbose=0,
        print_interval=500,
):
    """Compute a search_light.

    Parameters
    ----------
    X : array-like of shape at least 2D
        data to fit.

    y : array-like
        target variable to predict.

    estimator : estimator object implementing 'fit'
        object to use to fit the data

    A : scipy sparse matrix.
        adjacency matrix. Defines for each feature the neigbhoring features
        following a given structure of the data.

    groups : array-like, optional, (default None)
        group label for each sample for cross validation.

        .. note::
            This will have no effect for scikit learn < 0.18

    scoring : string or callable, optional
        The scoring strategy to use. See the scikit-learn documentation
        for possible values.
        If callable, it takes as arguments the fitted estimator, the
        test data (X_test) and the test target (y_test) if y is
        not None.

    cv : cross-validation generator, optional
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied.
    %(n_jobs_all)s
    %(verbose0)s

    Returns
    -------
    scores : array-like of shape (number of rows in A)
        search_light scores
    """
    group_iter = GroupIterator(len(A), n_jobs)
    with warnings.catch_warnings():  # might not converge
        warnings.simplefilter("ignore", ConvergenceWarning)
        scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(custom_group_iter_search_light)(
                [A[i] for i in list_i],
                estimator,
                X,
                y,
                groups,
                scoring,
                cv,
                thread_id + 1,
                len(A),
                print_interval,
            )
            for thread_id, list_i in enumerate(group_iter)
        )
    return np.concatenate(scores)


def pairwise_acc_captions(latents, predictions):
    return pairwise_accuracy(latents[INDICES_TEST_STIM_CAPTION], predictions[INDICES_TEST_STIM_CAPTION])


def pairwise_acc_images(latents, predictions):
    return pairwise_accuracy(latents[INDICES_TEST_STIM_IMAGE], predictions[INDICES_TEST_STIM_IMAGE])


def pairwise_acc(latents, predictions):
    assert len(latents) == len(predictions) == NUM_TEST_STIMULI
    return pairwise_accuracy(latents, predictions, IDS_TEST_STIM)


def run(args):
    for subject in args.subjects:
        train_fmri = dict()
        train_fmri['left'] = pickle.load(open(os.path.join(SURFACE_LEVEL_FMRI_DIR, f"{subject}_left_{args.resolution}_train.p"), 'rb'))
        train_fmri['right'] = pickle.load(open(os.path.join(SURFACE_LEVEL_FMRI_DIR, f"{subject}_right_{args.resolution}_train.p"), 'rb'))

        test_fmri = dict()
        test_fmri['left'] = pickle.load(open(os.path.join(SURFACE_LEVEL_FMRI_DIR, f"{subject}_left_{args.resolution}_test.p"), 'rb'))
        test_fmri['right'] = pickle.load(open(os.path.join(SURFACE_LEVEL_FMRI_DIR, f"{subject}_right_{args.resolution}_test.p"), 'rb'))

        train_stim_ids = pickle.load(open(os.path.join(SURFACE_LEVEL_FMRI_DIR, f"{subject}_stim_ids_train.p"), 'rb'))
        train_stim_types = pickle.load(open(os.path.join(SURFACE_LEVEL_FMRI_DIR, f"{subject}_stim_types_train.p"), 'rb'))

        test_stim_ids = pickle.load(open(os.path.join(SURFACE_LEVEL_FMRI_DIR, f"{subject}_stim_ids_test.p"), 'rb'))
        test_stim_types = pickle.load(open(os.path.join(SURFACE_LEVEL_FMRI_DIR, f"{subject}_stim_types_test.p"), 'rb'))

        for training_mode in args.training_modes:
            for model_name in args.models:
                model_name = model_name.lower()

                for features in args.features:
                    if features == FEATS_SELECT_DEFAULT:
                        features = get_default_features(model_name)

                    print(f"\nTRAIN MODE: {training_mode} | SUBJECT: {subject} | "
                          f"MODEL: {model_name} | FEATURES: {features}")

                    train_data_latents, nn_latent_transform = get_nn_latent_data(model_name, features,
                                                                                 args.vision_features,
                                                                                 train_stim_ids,
                                                                                 train_stim_types,
                                                                                 subject,
                                                                                 training_mode)
                    test_data_latents, _ = get_nn_latent_data(model_name, features, args.vision_features,
                                                              test_stim_ids,
                                                              test_stim_types,
                                                              subject,
                                                              "test",
                                                              nn_latent_transform=nn_latent_transform)
                    latents = np.concatenate((train_data_latents, test_data_latents))

                    fsaverage = datasets.fetch_surf_fsaverage(mesh=args.resolution)
                    for hemi in args.hemis:
                        print("Hemisphere: ", hemi)
                        if training_mode == "train_captions":
                            train_fmri_hemi = train_fmri[hemi][train_stim_types == 'caption']
                        elif training_mode == "train_images":
                            train_fmri_hemi = train_fmri[hemi][train_stim_types == 'image']
                        else:
                            train_fmri_hemi = train_fmri[hemi]

                        print(f"train_fmri_hemi shape: {train_fmri_hemi.shape}")
                        print(f"test_fmri_hemi shape: {test_fmri[hemi].shape}")

                        train_ids = list(range(len(train_fmri_hemi)))
                        test_ids = list(range(len(train_fmri_hemi), len(train_fmri_hemi) + len(test_fmri[hemi])))

                        X = np.concatenate((train_fmri_hemi, test_fmri[hemi]))

                        results_dir = get_results_dir(args, features, hemi, model_name, subject, training_mode)
                        results_file_name = f"alpha_{args.l2_regularization_alpha}.p"

                        nan_locations = np.isnan(X[0])
                        print(f"nan_locations: {nan_locations.sum()}")
                        assert np.all(nan_locations == np.isnan(X[-1]))
                        X = X[:, ~nan_locations]

                        infl_mesh = fsaverage[f"infl_{hemi}"]
                        coords, _ = surface.load_surf_mesh(infl_mesh)
                        coords = coords[~nan_locations]

                        nn = neighbors.NearestNeighbors(radius=args.radius)
                        results_dict = {}
                        results_dict["nan_locations"] = nan_locations
                        if args.radius is not None:
                            adjacency = [np.argwhere(arr == 1)[:, 0] for arr in nn.fit(coords).radius_neighbors_graph(coords).toarray()]
                            n_neighbors = [len(adj) for adj in adjacency]
                            results_dict["n_neighbors"] = n_neighbors
                            print(f"Number of neighbors within {args.radius}mm radius: {np.mean(n_neighbors):.1f} (max: {np.max(n_neighbors):.0f} | min: {np.min(n_neighbors):.0f})")
                        elif args.n_neighbors is not None:
                            distances, adjacency = nn.fit(coords).kneighbors(coords, n_neighbors=args.n_neighbors)
                            results_dict["distances"] = distances
                            print(f"Max distance among {args.n_neighbors} neighbors: {distances.max():.2f}mm")
                            print(f"Mean distance among {args.n_neighbors} neighbors: {distances.mean():.2f}mm")
                            print(f"Mean max distance: {distances.max(axis=1).mean():.2f}mm")

                        else:
                            raise RuntimeError("Need to set either radius or n_neighbors arg!")

                        model = make_pipeline(StandardScaler(), Ridge(alpha=args.l2_regularization_alpha))
                        pairwise_acc_scorers = {name: make_scorer(measure, greater_is_better=True) for name, measure
                                                in zip(["overall", "captions", "images"],
                                                       [pairwise_acc, pairwise_acc_captions, pairwise_acc_images])}
                        cv = [(train_ids, test_ids)]

                        start = time.time()
                        scores = custom_search_light(X, latents, estimator=model, A=adjacency, cv=cv, n_jobs=args.n_jobs,
                                              scoring=pairwise_acc_scorers, verbose=1, print_interval=500)
                        end = time.time()
                        print(f"Searchlight time: {int(end - start)}s")
                        test_scores = [score["test_overall"] for score in scores]
                        print(f"Mean score: {np.mean(test_scores):.2f} | Max score: {np.max(test_scores):.2f}")

                        results_dict["scores"] = scores
                        pickle.dump(results_dict, open(os.path.join(results_dir, results_file_name), 'wb'))


def get_results_dir(args, features, hemi, model_name, subject, training_mode):
    results_dir = os.path.join(SEARCHLIGHT_OUT_DIR, training_mode, model_name, features,
                               subject,
                               args.resolution, hemi)
    if args.radius is not None:
        results_dir = os.path.join(results_dir, f"radius_{args.radius}")
    else:
        results_dir = os.path.join(results_dir, f"n_neighbors_{args.n_neighbors}")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--training-modes", type=str, nargs="+", default=['train'],
                        choices=TRAIN_MODE_CHOICES)

    parser.add_argument("--models", type=str, nargs='+', default=['vilt'])
    parser.add_argument("--features", type=str, nargs='+', default=[FEATS_SELECT_DEFAULT],
                        choices=FEATURE_COMBINATION_CHOICES)
    parser.add_argument("--vision-features", type=str, default=VISION_MEAN_FEAT_KEY,
                        choices=VISION_FEAT_COMBINATION_CHOICES)

    parser.add_argument("--subjects", type=str, nargs='+', default=DEFAULT_SUBJECTS)

    parser.add_argument("--resolution", type=str, default="fsaverage7")

    parser.add_argument("--hemis", type=str, nargs="+", default=["left", "right"])

    parser.add_argument("--l2-regularization-alpha", type=float, default=1)

    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--n-neighbors", type=int, default=None)

    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(SEARCHLIGHT_OUT_DIR, exist_ok=True)

    run(args)
