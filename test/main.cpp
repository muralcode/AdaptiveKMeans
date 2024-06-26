//Copyright (c) 2024 Arithoptix Pty Ltd.
#include "AdaptiveKMeans.h"
#include <iostream>
#include <vector>

int main() {
    // Example data points (2D points)
    std::vector<Point> data = {
        {{1, 2}},
        {{1, 4}},
        {{1, 0}},
        {{4, 2}},
        {{4, 4}},
        {{4, 0}}
    };

    // Parameters
    int k = 2; // Number of clusters
    int maxIterations = 100; // Maximum number of iterations

    // Create AdaptiveKMeans object and perform clustering with Euclidean distance
    AdaptiveKMeans kmeans(data, k);
    std::vector<int> clusterAssignments = kmeans.cluster(maxIterations);

    // Output cluster assignments
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << "Data point " << i << " is assigned to cluster " << clusterAssignments[i] << std::endl;
    }

    // Perform clustering with Mahalanobis distance
    clusterAssignments = kmeans.cluster(maxIterations, true);

    // Output cluster assignments with Mahalanobis distance
    std::cout << "With Mahalanobis distance:" << std::endl;
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << "Data point " << i << " is assigned to cluster " << clusterAssignments[i] << std::endl;
    }

    return 0;
}
