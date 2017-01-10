#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "boost/filesystem.hpp"
#include "mosaic.h"
#include <iostream>
#include <algorithm>

namespace fs = boost::filesystem;
using std::cout;
using std::clog;
using std::cerr;
using std::endl;

using namespace mos;


namespace mos
{
    bool Mosaic::_is_image_file(const fs::path& path)
    {
        if (fs::is_regular_file(path) && path.extension() == ".jpg")
            return true;
        else
            return false;
    }

    void Mosaic::_crop_image(cv::Mat& output, const cv::Mat& image)
    {
        size_t size = std::min(image.rows, image.cols);
        size_t pad_height = image.rows - size;
        size_t pad_width = image.cols - size;

        output = image(cv::Rect(pad_width / 2, pad_height / 2, size, size));
    }

    void Mosaic::_load_image(cv::Mat& image, const fs::path& image_path)
    {
        try
        {
            if (!_is_image_file(image_path))
                throw InvalidImageFileException();

            image = cv::imread(image_path.string());

            if (image.empty())
                throw EmptyImageException();
        }
        catch (const InvalidImageFileException& e)
        {
            cerr << e.what() << ": " << image_path << endl;
        }
        catch (const EmptyImageException& e)
        {
            cerr << e.what() << ": " << image_path << endl;
        }
        catch (const std::exception& e)
        {
            cerr << e.what() << endl;
        }
    }

    void Mosaic::_compute_mean_color(cv::Mat& output, const cv::Mat& image,
                                     const size_t& step)
    {
        output.create(image.rows / step, image.cols / step, CV_32FC3);

        for (size_t i = 0; i < output.rows; ++i)
        {
            cv::Vec3f* row_p = output.ptr<cv::Vec3f>(i);
            for (size_t j = 0; j < output.cols; ++j)
            {
                cv::Scalar mean = cv::mean(image(cv::Rect(j * step, i * step,
                                                          step, step)));
                // TODO optimize
                row_p[j][0] = mean[0];
                row_p[j][1] = mean[1];
                row_p[j][2] = mean[2];
            }
        }
    }

    float Mosaic::_compute_similarity(const cv::Vec3f& input1,
                                      const cv::Mat& input2)
    {
        const cv::Vec3f* row_p = input2.ptr<cv::Vec3f>(0);
        float nominator = cv::sum(input1.mul(row_p[0]))[0];
        float denominator = cv::norm(input1) * cv::norm(row_p[0]);

        return nominator / denominator;
    }

    size_t Mosaic::_find_similar(std::vector<bool>& images_used,
                                 const cv::Vec3f& input,
                                 const std::vector<cv::Mat>& database)
    {
        size_t idx = 0;
        float highest_similarity = -1.f;
        int sum = 0;

        for (size_t i = 0; i < database.size(); ++i)
        {
            sum += images_used[i];
            if (images_used[i])
                continue;

            float similarity = _compute_similarity(input, database[i]);
            if (similarity > highest_similarity)
            {
                highest_similarity = similarity;
                idx = i;
            }
        }

        if (sum == database.size() - 1)
            images_used.assign(database.size(), false);
        else
            images_used[idx] = true;

        return idx;
    }

    void Mosaic::_mosaicing(const cv::Mat& base_feature,
                            const std::vector<cv::Mat>& images,
                            const std::vector<cv::Mat>& features,
                            const size_t& step,
                            const fs::path& output_path)
    {
        if (base_feature.empty() || features.empty())
            return;

        cv::Mat output(base_feature.rows * step, base_feature.cols * step,
                       CV_8UC3);
        std::vector<bool> images_used(images.size(), false);
        std::vector<size_t> ids(base_feature.rows * base_feature.cols);
        const cv::Vec3f* row_p = base_feature.ptr<cv::Vec3f>(0);
        cv::RNG random(0xFFFFFFFF);

        for (size_t i = 0; i < ids.size(); ++i)
            ids[i] = i;

        for (size_t i = 0; i < ids.size(); ++i)
        {
            size_t temp_id = random.uniform(0, ids.size() - i);
            size_t id  = ids[temp_id];
            ids[temp_id] = ids[ids.size() - 1 - i];

            size_t image_id = _find_similar(images_used, row_p[id], features);
            size_t height = id / base_feature.cols * step;
            size_t width = id % base_feature.cols * step;
            images[image_id].copyTo(output(cv::Rect(width, height,
                                                    step, step)));
        }

        cv::imwrite(output_path.string(), output);
        clog << "Finish creating mosaic: " << output_path << endl;
    }

    void Mosaic::_prepare_base(cv::Mat& output, const fs::path& image_path,
                               const size_t& tile_size)
    {
        cv::Mat image;
        _load_image(image, image_path);
        _compute_mean_color(output, image, tile_size);
    }

    void Mosaic::_prepare_features_database(std::vector<cv::Mat>& images,
                                            std::vector<cv::Mat>& features,
                                            const fs::path& images_path,
                                            const size_t& tile_size)
    {
        try
        {
            fs::directory_iterator end_it;
            for (fs::directory_iterator it(images_path); it != end_it; ++it)
            {
                cv::Mat image;
                _load_image(image, it->path());
                if (!image.empty())
                {
                    features.push_back(cv::Mat());
                    images.push_back(cv::Mat());

                    _crop_image(image, image);
                    _compute_mean_color(features.back(), image, image.cols);
                    cv::resize(image, images.back(),
                               cv::Size(tile_size, tile_size));
                }
            }

            if (images.empty())
                throw EmptyImageDatabaseException();
        }
        catch (const EmptyImageDatabaseException& e)
        {
            cerr << e.what() << ": " << images_path << endl;
        }
        catch (const std::exception& e)
        {
            cerr << e.what() << endl;
        }
    }

    void Mosaic::create(const size_t& tile_size,
                        const fs::path& base_image_path,
                        const fs::path& images_path,
                        const fs::path& output_path)
    {
        cv::Mat base_feature;
        std::vector<cv::Mat> images, features;

        _prepare_base(base_feature, base_image_path, tile_size);
        _prepare_features_database(images, features, images_path, tile_size);

        _mosaicing(base_feature, images, features, tile_size, output_path);
    }
}
