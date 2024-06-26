//Copyright (c) 2024 Arithoptix Pty Ltd.
#ifndef ADAPTIVEKMEANS_H
#define ADAPTIVEKMEANS_H

#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>

// Define a structure for a data point
struct Point {
    std::vector<double> coords;
};

// custom Adaptive K-means clustering class
class AdaptiveKMeans {
public:
    AdaptiveKMeans(const std::vector<Point>& data, int k);
    std::vector<int> cluster(int maxIterations = 100, bool useMahalanobis = false);

    // Ensemble clustering
    static std::vector<int> ensembleClustering(const std::vector<std::vector<int>>& clusteringResults);

private:
    std::vector<Point> data_;
    int k_;
    std::vector<Point> centroids_;

    // Private functions
    double calculateDistance(const Point& p1, const Point& p2, bool useMahalanobis);
    Eigen::MatrixXd calculateCovarianceMatrix();
    std::vector<Point> initializeCentroids(const std::vector<Point>& data, int k);
    std::vector<Point> initializeCentroidsRobust(const std::vector<Point>& data, int k, int numInitializationAttempts);
    void featureScaling();
};

#endif // ADAPTIVEKMEANS_H
