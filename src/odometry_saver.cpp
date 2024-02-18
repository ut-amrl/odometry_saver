#include <Eigen/Core>
#include <Eigen/Geometry>
#include <atomic>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

#include <filesystem>

#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <tf/transform_listener.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

template <typename T> void save_data(const std::string &dst_directory, const T &data);

template <>
void save_data(const std::string &dst_directory,
               const sensor_msgs::PointCloud2ConstPtr &data) {
  std::stringstream dst_filename;
  dst_filename << dst_directory << "/" << data->header.stamp.sec << "_" << std::setw(9)
               << std::setfill('0') << data->header.stamp.nsec << ".pcd";

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::fromROSMsg(*data, *cloud);

  pcl::io::savePCDFileBinary(dst_filename.str(), *cloud);
}

template <>
void save_data(const std::string &dst_directory,
               const nav_msgs::OdometryConstPtr &data) {
  std::stringstream dst_filename;
  dst_filename << dst_directory << "/" << data->header.stamp.sec << "_" << std::setw(9)
               << std::setfill('0') << data->header.stamp.nsec << ".odom";

  const auto &pose = data->pose.pose;

  Eigen::Isometry3d odom = Eigen::Isometry3d::Identity();
  odom.translation() =
      Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  odom.linear() = Eigen::Quaterniond(pose.orientation.w, pose.orientation.x,
                                     pose.orientation.y, pose.orientation.z)
                      .normalized()
                      .toRotationMatrix();

  std::ofstream ofs(dst_filename.str());
  ofs << odom.matrix();
  ofs.close();
}

template <typename T> class SaveQueue {
public:
  SaveQueue(const std::string &dst_directory)
      : kill_switch(false), dst_directory(dst_directory) {
    thread = std::thread([this]() { save_task(); });
  }

  size_t size() {
    std::lock_guard<std::mutex> lock(mutex);
    return queue.size();
  }

  void push(const T &data) {
    std::lock_guard<std::mutex> lock(mutex);
    queue.push_back(data);
    data_available.notify_one();
  }

private:
  void save_task() {
    std::unique_lock<std::mutex> lock(mutex);
    while (!kill_switch) {
      data_available.wait(lock, [this] { return !queue.empty() || kill_switch; });

      if (kill_switch && queue.empty()) {
        break;
      }

      T data = queue.front();
      queue.pop_front();

      lock.unlock();
      save_data(dst_directory, data);
      lock.lock();
    }
  }

private:
  std::atomic_bool kill_switch;

  std::mutex mutex;
  std::condition_variable data_available;
  std::deque<T> queue;

  std::thread thread;

  std::string dst_directory;
};

class OdometrySaverNode {
public:
  OdometrySaverNode()
      : nh("~"), endpoint_frame(nh.param<std::string>("endpoint_frame", "base_link")),
        origin_frame(nh.param<std::string>("origin_frame", "map")),
        dst_directory(nh.param<std::string>("dst_directory", "/tmp/odometry")),
        saved_points(0), saved_odometry(0), points_save_queue(dst_directory),
        odometry_save_queue(dst_directory),
        points_sub(nh.subscribe<sensor_msgs::PointCloud2>(
            "/points", 128, &OdometrySaverNode::points_callback, this)),
        odometry_sub(nh.subscribe<nav_msgs::Odometry>(
            "/odom", 128, &OdometrySaverNode::odometry_callback, this)),
        tf_listener(ros::DURATION_MAX) {
    std::filesystem::create_directories(dst_directory);

    timer = nh.createWallTimer(ros::WallDuration(1.0),
                               &OdometrySaverNode::timer_callback, this);

    std::string file_path = dst_directory + "/pose.txt";
    std::cout << "pose file path: " << file_path << std::endl;
    if (std::filesystem::exists(file_path)) {
      std::filesystem::remove(file_path);
    }
    pose_file_.open(file_path, std::ios::out | std::ios::app);
    if (!pose_file_.is_open()) {
      ROS_ERROR_STREAM("failed to open file: " << file_path);
    }
  }

  ~OdometrySaverNode() {
    if (pose_file_.is_open()) {
      pose_file_.close();
    }
  }

private:
  void timer_callback(const ros::WallTimerEvent &e) {
    std::cout << "--- saver queues ---" << std::endl;

    ROS_INFO_STREAM("queue points:" << points_save_queue.size()
                                    << "  odometry:" << odometry_save_queue.size());
    ROS_INFO_STREAM("saved points:" << saved_points << "  odometry:" << saved_odometry);
  }

  void points_callback(const sensor_msgs::PointCloud2ConstPtr &points_msg) {
    saved_points++;
    points_save_queue.push(points_msg);
  }

  void odometry_callback(const nav_msgs::OdometryConstPtr &odometry_msg) {
    saved_odometry++;
    Eigen::Matrix4d origin2odom =
        lookup_eigen(odometry_msg->header.frame_id, origin_frame);
    Eigen::Matrix4d odom2base =
        lookup_eigen(endpoint_frame, odometry_msg->child_frame_id);

    const auto &pose = odometry_msg->pose.pose;
    Eigen::Matrix4d odombase2odom = Eigen::Matrix4d::Identity();
    odombase2odom.block<3, 1>(0, 3) =
        Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
    odombase2odom.block<3, 3>(0, 0) =
        Eigen::Quaterniond(pose.orientation.w, pose.orientation.x, pose.orientation.y,
                           pose.orientation.z)
            .toRotationMatrix();

    Eigen::Matrix4d result = odom2base * odombase2odom * origin2odom;
    Eigen::Vector3d t(result.block<3, 1>(0, 3));
    Eigen::Quaterniond q(result.block<3, 3>(0, 0));

    // Save transformed odometry
    nav_msgs::OdometryPtr transformed_odometry_msg(new nav_msgs::Odometry());
    *transformed_odometry_msg = *odometry_msg;
    transformed_odometry_msg->header.frame_id = origin_frame;
    transformed_odometry_msg->child_frame_id = endpoint_frame;
    transformed_odometry_msg->pose.pose.position.x = t.x();
    transformed_odometry_msg->pose.pose.position.y = t.y();
    transformed_odometry_msg->pose.pose.position.z = t.z();
    transformed_odometry_msg->pose.pose.orientation.w = q.w();
    transformed_odometry_msg->pose.pose.orientation.x = q.x();
    transformed_odometry_msg->pose.pose.orientation.y = q.y();
    transformed_odometry_msg->pose.pose.orientation.z = q.z();

    odometry_save_queue.push(transformed_odometry_msg);

    // Save pose to file
    if (pose_file_.is_open()) {
      pose_file_ << std::fixed << std::setprecision(6)
                 << odometry_msg->header.stamp.toSec() << " " << std::fixed
                 << std::setprecision(8) << t.x() << " " << t.y() << " " << t.z() << " "
                 << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << std::endl;
    }
  }

  Eigen::Matrix4d lookup_eigen(const std::string &target, const std::string &source,
                               const ros::Time &stamp = ros::Time(0)) {
    if (!tf_listener.waitForTransform(target, source, stamp, ros::Duration(5.0))) {
      ROS_WARN_STREAM("failed to lookup transform from " << source << " to " << target);
      return Eigen::Matrix4d::Identity();
    }

    tf::StampedTransform transform;
    tf_listener.lookupTransform(target, source, stamp, transform);

    Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
    transform.getOpenGLMatrix(matrix.data());

    return matrix;
  }

private:
  ros::NodeHandle nh;
  ros::WallTimer timer;

  std::string endpoint_frame;
  std::string origin_frame;

  int saved_points;
  int saved_odometry;

  std::string dst_directory;
  SaveQueue<sensor_msgs::PointCloud2ConstPtr> points_save_queue;
  SaveQueue<nav_msgs::OdometryConstPtr> odometry_save_queue;
  std::ofstream pose_file_;

  ros::Subscriber points_sub;
  ros::Subscriber odometry_sub;

  tf::TransformListener tf_listener;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "odometry_saver");

  OdometrySaverNode node;

  ros::spin();

  return 0;
}
