import argparse
import pandas as pd
import pickle
import time
from src.supervised_learning.utils.welding_arc_system.generate_training_data_RLC import DataGenerator
from sktime.classification.compose import TimeSeriesForestClassifier
from sktime.utils.data_container import detabularize
from sktime.transformers.series_as_features.summarize import RandomIntervalFeatureExtractor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sktime.utils.time_series import time_series_slope
from joblib import parallel_backend
import numpy as np


def main():
    generator = DataGenerator(labeled_data_file=args.labeled_data_file, data_util_file=args.data_util_file,
                              threshold=args.threshold, dt=args.dt, L=args.L, tmin=args.tmin, tmax=args.tmax)
    training_data, test_data = generator.get_data(ts_nth_element=args.ts_nth_element,
                                                                   training_frac=0.7)
    steps = [
        ('extract', RandomIntervalFeatureExtractor(n_intervals='sqrt',
                                                   features=[np.mean, np.std, time_series_slope])),
        ('clf', DecisionTreeClassifier())
    ]
    time_series_tree = Pipeline(steps)
    tsf = TimeSeriesForestClassifier(
        estimator=time_series_tree,
        n_estimators=args.n_estimators,
        criterion='entropy' if args.criterion == 'entropy' else 'gini',
        bootstrap=True,
        oob_score=True,
        random_state=1,
        # n_jobs=4,
        verbose=1
    )
    x = detabularize(pd.DataFrame(training_data[:,1:]))
    try:
        with parallel_backend('threading', n_jobs=args.n_jobs):
            tsf = tsf.fit(x, training_data[:,0])
        with open('{save_file_name}.pickle'.format(save_file_name=args.save_file_name), 'wb') \
                as TimeSeriesForestModel:
            pickle.dump(tsf, TimeSeriesForestModel, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--L", type=float, default=1.0)
    p.add_argument("--tmin", type=int, default=1800)
    p.add_argument("--tmax", type=int, default=2000)
    p.add_argument("--ts_nth_element", type=int, default=8)
    p.add_argument("--threshold", type=int, default=23)
    p.add_argument("--n_jobs", type=int, default=20)
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--criterion", type=str, default="entropy")
    p.add_argument("--labeled_data_file", type=str, default="labeled_data_file.txt")
    p.add_argument("--data_util_file", type=str, default="data_util_file.txt")
    p.add_argument("--save_file_name", type=str, default="TimeSeriesForestModel")
    args = p.parse_args()
    start = time.time()
    main()
    end = time.time()
    print(str(end - start))