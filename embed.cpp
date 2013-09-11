#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>
#include "embed.h"

using namespace std;
using namespace boost;
using namespace aaalgo;

void read_data (string const &path, bool binary, vector<Embed::Entry> *data) {
    cerr << "Reading " << (binary ? "binary" : "text") << " data from " << path << " ... ";
    if (binary) {
        ifstream is(path.c_str(), ios::binary);
        is.seekg(0, ios::end);
        size_t sz = is.tellg();
        BOOST_VERIFY(sz % sizeof(Embed::Entry) == 0);
        data->resize(sz / sizeof(Embed::Entry));
        is.seekg(0, ios::beg);
        is.read(reinterpret_cast<char *>(&data->at(0)), sz);
    }
    else {
        ifstream is(path.c_str());
        Embed::Entry r;
        while (is >> r.row >> r.col >> r.value) {
            data->push_back(r);
        }
    }
    cerr << data->size() << " items. OK." << endl;
}

int main (int argc, char *argv[]) {
    int maxit;
    float th;
    Embed::Options options;
    std::string train_path;
    std::string test_path;
    std::string load_path;
    std::string save_path;
    bool binary = false;
    bool symmetric = false;
    bool override = false;


    namespace po = boost::program_options; 
    
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("train", po::value(&train_path), "training data")
    ("test", po::value(&test_path), "testing data")
    ("save", po::value(&save_path), "save model file")
    ("load", po::value(&load_path), "load model file")
    ("binary", "use binary")
    ("symmetric", "symmetric")
    ("override", "override loaded parameters with command")
    ("maxit", po::value(&maxit)->default_value(0), "max iterations.")
    ("th", po::value(&th)->default_value(0), "stopping MSRE.")
    ;

    options.add(desc.add_options(), "");

    po::positional_options_description p;

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help") || (vm.count("train") == 0 && vm.count("load") == 0)) {
        std::cerr << "Usage:" << endl;
        std::cerr << "\t" << argv[0] << " [options] --input FILE and/or --load FILE" << endl;
        std::cerr << desc;
        return 1;
    }

    if (vm.count("binary")) binary = true;
    if (vm.count("symmetric")) symmetric = true;
    if (vm.count("override")) override = true;

    Embed embed;
    if (!load_path.empty()) {
        cerr << "Loading model " << load_path << " ... ";
        embed.load(load_path);
        cerr << "OK." << endl;
    }

    if (override || load_path.empty()) {
        embed.options(options);
    }

    if (!train_path.empty()) {
        vector<Embed::Entry> data;
        read_data(train_path, binary, &data);
        if (load_path.empty()) {
            int row = -1, col = -1;
            for (auto const &r: data) {
                //cerr << r.row << ' ' << r.col << endl;
                if (r.row > row) row = r.row;
                if (r.col > col) col = r.col;
            }
            ++row;
            ++col;
            if (symmetric) {
                if (col > row) row = col;
                col = 0;
                cerr << row << " rows." << endl;
            }
            else {
                cerr << row << " rows, " << col << " columns." << endl;
            }
            embed.init(row, col, data);
        }
        cerr << "Training ..." << endl;
        random_shuffle(data.begin(), data.end());
            //init(s1, s2, data);
        for (int it = 0; maxit == 0 || it < maxit; ++it) {
            float v = embed.loop(data);
            cerr << it << '\t' << v << endl;
            if (v < th) break;
        }
    }

    if (!save_path.empty()) {
        embed.save(save_path);
    }

    if (!test_path.empty()) {
        vector<Embed::Entry> data;
        read_data(test_path, binary, &data);
        vector<float> output(data.size());
        float err = 0;
#pragma omp parallel for reduction(+:err)
        for (unsigned i = 0; i < data.size(); ++i) {
            float v = embed.predict(data[i].row, data[i].col);
            v -= data[i].value;
            err += v * v;
        }
        err /= data.size();
        err = sqrt(err);
        cerr << "RMSE: " << err << endl;
    }
    return 0;
}

