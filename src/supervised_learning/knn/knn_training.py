import argparse
import pandas as pd
import pickle
import time
from src.supervised_learning.utils.welding_arc_system.generate_training_data_RLC import DataGenerator
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.utils.data_container import detabularize
from joblib import parallel_backend


def main():
    generator = DataGenerator(labeled_data_file=args.labeled_data_file, data_util_file=args.data_util_file,
                              threshold=args.threshold, dt=args.dt, L=args.L, tmin=args.tmin, tmax=args.tmax)
    training_data, test_data = generator.get_data(ts_nth_element=args.ts_nth_element,
                                                                   training_frac=0.7)
    knn = KNeighborsTimeSeriesClassifier(n_neighbors=args.n_neighbors, verbose=1, metric="dtw")
    x = detabularize(pd.DataFrame(training_data[:,1:]))
    try:
        with parallel_backend('threading', n_jobs=args.n_jobs):
            knn = knn.fit(x, training_data[:,0])
        with open('{save_file_name}.pickle'.format(save_file_name=args.save_file_name), 'wb') \
                as KNeighborsTimeSeriesModel:
            pickle.dump(knn, KNeighborsTimeSeriesModel, protocol=pickle.HIGHEST_PROTOCOL)
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
    p.add_argument("--n_neighbors", type=int, default=1)
    p.add_argument("--labeled_data_file", type=str, default="labeled_data_file.txt")
    p.add_argument("--data_util_file", type=str, default="data_util_file.txt")
    p.add_argument("--save_file_name", type=str, default="KNeighborsTimeSeriesModel")
    args = p.parse_args()
    start = time.time()
    main()
    end = time.time()
    print(str(end - start))