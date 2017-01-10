#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
#include "mosaic.h"
#include <iostream>

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using std::cout;
using std::clog;
using std::cerr;
using std::endl;


int main(int argc, char * argv[])
{
    fs::path base_image_path, images_path, output_path;
    try
    {
        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h",
             "display the help message")
            ("base_image,i", po::value<fs::path>(&base_image_path)->required(),
             "path to base image")
            ("images_dataset,d", po::value<fs::path>(&images_path)->required(),
             "path to images dataset")
            ("output,o", po::value<fs::path>(&output_path)->required(),
             "path to output image")
            ;
        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
        if (vm.count("help")) {
            cout << "Usage: main [options]\n";
            cout << desc;
            return EXIT_SUCCESS;
        }
        po::notify(vm);
    }
    catch (const po::error& e)
    {
        cerr << "Error while parsing command-line arguments: " \
             << e.what() << "\n" \
             << "Use --help to display a list of options." << endl;
        return EXIT_SUCCESS;
    }

    mos::Mosaic mosaic;
    mosaic.create(8, base_image_path, images_path, output_path);

    return EXIT_SUCCESS;
}
