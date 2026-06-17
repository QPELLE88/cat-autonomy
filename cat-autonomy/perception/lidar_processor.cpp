/**
 * CAT Autonomous Trucks — LiDAR Point Cloud Processor
 * Real-time clustering and preprocessing for obstacle detection pipeline.
 * 
 * Processes raw Velodyne VLP-32C and Ouster OS1-128 data streams,
 * performs ground plane removal, and outputs clustered point clouds
 * to the Python obstacle detection layer.
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>

namespace cat_autonomy {
namespace perception {

struct Point3D {
    float x, y, z;
    float intensity;
    uint32_t timestamp_ns;
    uint8_t sensor_id;
};

struct PointCluster {
    std::vector<Point3D> points;
    float centroid_x, centroid_y, centroid_z;
    float bbox_min_x, bbox_min_y, bbox_min_z;
    float bbox_max_x, bbox_max_y, bbox_max_z;
};

class LiDARProcessor {
public:
    LiDARProcessor(float ground_threshold = 0.3f,
                   float cluster_distance = 0.5f,
                   int min_cluster_size = 10)
        : ground_threshold_(ground_threshold),
          cluster_distance_(cluster_distance),
          min_cluster_size_(min_cluster_size) {}

    /**
     * Remove ground plane points using RANSAC-based estimation.
     * Returns non-ground points for obstacle detection.
     */
    std::vector<Point3D> removeGroundPlane(const std::vector<Point3D>& raw_points) {
        std::vector<Point3D> non_ground;
        non_ground.reserve(raw_points.size() / 2);

        // Simplified ground removal: z-threshold relative to sensor mount
        // In production, use RANSAC plane fitting
        for (const auto& pt : raw_points) {
            if (pt.z > ground_threshold_) {
                non_ground.push_back(pt);
            }
        }
        return non_ground;
    }

    /**
     * Euclidean clustering of non-ground points.
     * Groups nearby points into obstacle candidates.
     */
    std::vector<PointCluster> clusterPoints(const std::vector<Point3D>& points) {
        std::vector<PointCluster> clusters;
        std::vector<bool> processed(points.size(), false);

        for (size_t i = 0; i < points.size(); ++i) {
            if (processed[i]) continue;

            PointCluster cluster;
            std::vector<size_t> queue = {i};
            processed[i] = true;

            while (!queue.empty()) {
                size_t idx = queue.back();
                queue.pop_back();
                cluster.points.push_back(points[idx]);

                // Find neighbors within cluster_distance_
                for (size_t j = 0; j < points.size(); ++j) {
                    if (processed[j]) continue;
                    float dx = points[idx].x - points[j].x;
                    float dy = points[idx].y - points[j].y;
                    float dz = points[idx].z - points[j].z;
                    float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                    if (dist <= cluster_distance_) {
                        processed[j] = true;
                        queue.push_back(j);
                    }
                }
            }

            if (static_cast<int>(cluster.points.size()) >= min_cluster_size_) {
                computeClusterMetrics(cluster);
                clusters.push_back(std::move(cluster));
            }
        }
        return clusters;
    }

    /**
     * Full processing pipeline: ground removal -> clustering -> output.
     */
    std::vector<PointCluster> processFrame(const std::vector<Point3D>& raw_frame) {
        auto non_ground = removeGroundPlane(raw_frame);
        return clusterPoints(non_ground);
    }

private:
    float ground_threshold_;
    float cluster_distance_;
    int min_cluster_size_;

    void computeClusterMetrics(PointCluster& cluster) {
        float sum_x = 0, sum_y = 0, sum_z = 0;
        cluster.bbox_min_x = cluster.bbox_min_y = cluster.bbox_min_z = 1e9f;
        cluster.bbox_max_x = cluster.bbox_max_y = cluster.bbox_max_z = -1e9f;

        for (const auto& pt : cluster.points) {
            sum_x += pt.x; sum_y += pt.y; sum_z += pt.z;
            cluster.bbox_min_x = std::min(cluster.bbox_min_x, pt.x);
            cluster.bbox_min_y = std::min(cluster.bbox_min_y, pt.y);
            cluster.bbox_min_z = std::min(cluster.bbox_min_z, pt.z);
            cluster.bbox_max_x = std::max(cluster.bbox_max_x, pt.x);
            cluster.bbox_max_y = std::max(cluster.bbox_max_y, pt.y);
            cluster.bbox_max_z = std::max(cluster.bbox_max_z, pt.z);
        }

        float n = static_cast<float>(cluster.points.size());
        cluster.centroid_x = sum_x / n;
        cluster.centroid_y = sum_y / n;
        cluster.centroid_z = sum_z / n;
    }
};

} // namespace perception
} // namespace cat_autonomy
