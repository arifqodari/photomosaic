#ifndef _MOSAIC_LIB_H_
#define _MOSAIC_LIB_H_

#include "opencv2/core/core.hpp"
#include "boost/filesystem.hpp"

namespace fs = boost::filesystem;

namespace mos
{
    struct InvalidImageFileException : public std::exception
    {
        const char * what () const throw ()
        {
            return "Invalid image file";
        }
    };

    struct EmptyImageException : public std::exception
    {
        const char * what () const throw ()
        {
            return "Empty image";
        }
    };

    struct EmptyImageDatabaseException : public std::exception
    {
        const char * what () const throw ()
        {
            return "Empty image database";
        }
    };

    class Mosaic
    {
        bool _is_image_file(const fs::path& path);
        void _crop_image(cv::Mat& output, const cv::Mat& image);
        void _load_image(cv::Mat& image, const fs::path& image_path);
        void _compute_mean_color(cv::Mat& output, const cv::Mat& image,
                                 const size_t& step);
        float _compute_similarity(const cv::Vec3f& input1,
                                  const cv::Mat& input2);
        size_t _find_similar(std::vector<bool>& images_used,
                             const cv::Vec3f& input,
                             const std::vector<cv::Mat>& database);
        void _mosaicing(const cv::Mat& base_feature,
                        const std::vector<cv::Mat>& images,
                        const std::vector<cv::Mat>& features,
                        const size_t& step,
                        const fs::path& output_path);
        void _prepare_base(cv::Mat& output, const fs::path& image_path,
                           const size_t& tile_size);
        void _prepare_features_database(std::vector<cv::Mat>& images,
                                        std::vector<cv::Mat>& features,
                                        const fs::path& images_path,
                                        const size_t& tile_size);

        public:

        Mosaic() {};
        ~Mosaic() {};

        void create(const size_t& tile_size,
                    const fs::path& base_image_path,
                    const fs::path& images_path,
                    const fs::path& output_path);
    };
}


#endif // _MOSAIC_LIB_H_
