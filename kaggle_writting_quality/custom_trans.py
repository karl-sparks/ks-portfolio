import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans


class KMeansTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters, random_seed):
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_seed, n_init="auto"
        )
        self.feature_names_ = None

    def fit(self, X, y=None):
        self.kmeans_.fit(X=X, y=y)
        self.feature_names_ = [f"Cluster_{i}" for i in range(self.n_clusters)]
        return self

    def transform(self, X, y=None):
        clusters = self.kmeans_.transform(X)

        return np.hstack((X, clusters))

    def get_feature_names_out(self, names=None):
        if names is None:
            return self.feature_names_

        return np.concatenate([names, self.feature_names_])
