#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>
#include <boost/timer/timer.hpp>
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
    int every;
    float th;
    Embed::Options options;
    std::string train_path;
    std::string test_path;
    std::string load_path;
    std::string save_path;
    bool binary = false;
    bool symmetric = false;
    bool override = false;
    bool snapshot = false;
    bool reshuffle = false;


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
    ("every", po::value(&every)->default_value(10), "")
    ("snapshot", "")
    ("reshuffle", "")
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
    if (vm.count("snapshot")) snapshot = true;
    if (vm.count("reshuffle")) reshuffle = true;


    Embed embed;
    if (!load_path.empty()) {
        cerr << "Loading model " << load_path << " ... ";
        embed.load(load_path);
        cerr << "OK." << endl;
    }

    if (override || load_path.empty()) {
        embed.options(options);
    }

    vector<Embed::Entry> test_data;
    if (!test_path.empty()) {
        read_data(test_path, binary, &test_data);
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
        timer::cpu_timer timer;
        for (int it = 0; maxit == 0 || it < maxit; ++it) {
            float v = 0;
            {
                timer::auto_cpu_timer timer(cerr);
                if (reshuffle) {
                    random_shuffle(data.begin(), data.end());
                }
                v = embed.loop(data);
                cerr << it << '\t' << v << endl;
                cerr << "Loop elapsed: ";
            }
            cerr << "Total elapsed: " << timer::format(timer.elapsed());
            if (every > 0 && (it + 1) % every == 0) {
                if (snapshot) {
                    embed.save("snapshot." + save_path + "." + lexical_cast<string>((it + 1) / every));
                }
                if (test_data.size()) {
                    cerr << "RMSE: " << embed.evaluate(test_data);
                }
            }
            if (v < th) break;
        }
    }

    if (!save_path.empty()) {
        embed.save(save_path);
    }

    if (test_data.size()) {
        cerr << "RMSE: " << embed.evaluate(test_data) << endl;
    }
    return 0;
}

