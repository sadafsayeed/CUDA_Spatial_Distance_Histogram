# CUDA_Spatial_Distance_Histogram
This project is a CUDA-based implementation of Spatial Distance Histogram (SDH) computation for a set of 3D points. Given a collection of points and a user-defined bucket width w, the program calculates the number of distances falling into discrete distance ranges, or buckets, such as [0, w), [w, 2w), and so on.

Key features include:
  - Parallel Processing on GPU: Utilizes CUDA to compute distances between points and populate histogram buckets in parallel, where each GPU thread processes one data point.
  - Atomic Operations: Ensures thread safety using atomic operations to prevent concurrent modifications of shared memory when updating histogram counts.
  - Performance Comparison: Computes the SDH on both CPU and GPU, comparing the two histograms bucket by bucket to identify any discrepancies.
  - Flexible Execution: The program can be run with different numbers of CUDA blocks and block sizes, ensuring adaptability across different hardware configurations.
