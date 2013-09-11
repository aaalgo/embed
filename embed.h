#ifndef AAALGO_EMBED
#define AAALGO_EMBED

#include <random>
#include <boost/progress.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/program_options/options_description.hpp>

namespace aaalgo {

    using namespace std;
    using namespace boost;
    namespace ublas = boost::numeric::ublas;
    namespace po = boost::program_options;
    

    class Embed {
    public:
        typedef ublas::vector<float> vector;
        typedef ublas::scalar_vector<float> scalar_vector;
        typedef ublas::matrix<float> matrix;
        typedef ublas::matrix_row<matrix> matrix_row;

        class Options {
            int m_dim;
            float m_r1, m_r2;   // regulation terms
            float m_mom, m_eps, m_init;
            float m_min;
            friend class Embed;
        public:
            void add (po::options_description_easy_init init, string const &prefix = "embed") {
                init
                    ((prefix + "dim").c_str(), po::value(&m_dim)->default_value(10), "dimension")
                    ((prefix + "r1").c_str(), po::value(&m_r1)->default_value(0.1), "regulate")
                    ((prefix + "r2").c_str(), po::value(&m_r2)->default_value(0.1), "regulate")
                    ((prefix + "mom").c_str(), po::value(&m_mom)->default_value(0.9), "moment")
                    ((prefix + "eps").c_str(), po::value(&m_eps)->default_value(0.01), "")
                    ((prefix + "init").c_str(), po::value(&m_init)->default_value(0.1), "")
                    ((prefix + "min").c_str(), po::value(&m_min)->default_value(nanf("")), "")
                    //(prefix + "-max", po::value(&m_max)->default_value(nanf("")), "")
                ;
            }
        };

    private:
        int m_size1, m_size2;
        float m_min;
        mutable matrix m_data1;
        mutable matrix m_data2;
        vector m_bias1;
        vector m_bias2;

        matrix m_delta1;
        matrix m_delta2;
        vector m_bias_delta1;
        vector m_bias_delta2;

        Options m_options;

        template <typename T>
        static void write_ublas (ostream &os, T const &v) {
            os.write(reinterpret_cast<char const *>(&v.data()[0]), sizeof(v.data()[0]) * v.data().size());
        }

        template <typename T>
        static void read_ublas (istream &os, T &v) {
            os.read(reinterpret_cast<char *>(&v.data()[0]), sizeof(v.data()[0]) * v.data().size());
        }

    public:
        struct Entry {
            int row, col;
            float value;
        };

        Embed () {
        }

        void options (Options const &opt) {
            m_options = opt;
        }

        void save (string const &path) const {
            ofstream os(path.c_str(), ios::binary);
            os.write(reinterpret_cast<char const *>(&m_options), sizeof(m_options));
            os.write(reinterpret_cast<char const *>(&m_size1), sizeof(m_size1));
            os.write(reinterpret_cast<char const *>(&m_size2), sizeof(m_size2));
            os.write(reinterpret_cast<char const *>(&m_min), sizeof(m_min));
            write_ublas(os, m_data1);
            write_ublas(os, m_bias1);
            write_ublas(os, m_delta1);
            write_ublas(os, m_bias_delta1);
            if (m_size2) {
                write_ublas(os, m_data2);
                write_ublas(os, m_bias2);
                write_ublas(os, m_delta2);
                write_ublas(os, m_bias_delta2);
            }
        }

        void load (string const &path) {
            ifstream is(path.c_str(), ios::binary);
            is.read(reinterpret_cast<char *>(&m_options), sizeof(m_options));
            is.read(reinterpret_cast<char *>(&m_size1), sizeof(m_size1));
            is.read(reinterpret_cast<char *>(&m_size2), sizeof(m_size2));
            is.read(reinterpret_cast<char *>(&m_min), sizeof(m_min));
            m_data1.resize(m_size1, m_options.m_dim);
            m_delta1.resize(m_size1, m_options.m_dim);
            m_bias1.resize(m_size1);
            m_bias_delta1.resize(m_size1);
            read_ublas(is, m_data1);
            read_ublas(is, m_bias1);
            read_ublas(is, m_delta1);
            read_ublas(is, m_bias_delta1);
            if (m_size2) {
                m_data2.resize(m_size2, m_options.m_dim);
                m_delta2.resize(m_size2, m_options.m_dim);
                m_bias2.resize(m_size2);
                m_bias_delta2.resize(m_size2);
                read_ublas(is, m_data2);
                read_ublas(is, m_bias2);
                read_ublas(is, m_delta2);
                read_ublas(is, m_bias_delta2);
            }
        }

        void init (int s1, int s2, std::vector<Entry> const &data) {
            m_size1 = s1;
            m_size2 = s2;
            m_data1.resize(s1, m_options.m_dim);
            m_delta1.resize(s1, m_options.m_dim);
            m_bias1.resize(s1);
            m_bias_delta1.resize(s1);
            if (s2) {
                m_data2.resize(s2, m_options.m_dim);
                m_delta2.resize(s2, m_options.m_dim);
                m_bias2.resize(s2);
                m_bias_delta2.resize(s2);
            }

            // find minimal value
            if (isnan(m_options.m_min)) {
                m_min = data[0].value;
                for (auto const &e: data) {
                    if (e.value < m_min) {
                        m_min = e.value;
                    }
                }
                cerr << "FOUND MINIMAL VALUE: " << m_min << endl;
            }
            else {
                m_min = m_options.m_min;
            }

            float sum = 0;
            for (auto const &e: data) {
                sum += e.value;
            }
            sum /= data.size();
            sum -= m_min;
            float mean = sqrt(sum / m_options.m_dim);
            mt19937 random;
            normal_distribution<float> normal(0, mean * m_options.m_init);
            for (float &e: m_data1.data()) {
                e = normal(random);
            }
            for (float &e: m_data2.data()) {
                e = normal(random);
            }
            for (float &e: m_bias1.data()) {
                e = 0;
            }
            for (float &e: m_bias2.data()) {
                e = 0;
            }
        }
        /* Theory:
         *      min   0.5 * (Vij - Ui . Mj)^2 + 0.5 * Cu * ||Ui||^2 + 0.5 * Cm * ||Mj||^2
         *
         *
         *
         */
        float predict (int row, int col) const {
            if (m_size2) {
                return ublas::inner_prod(matrix_row(m_data1, row), matrix_row(m_data2, col)) + m_bias1(row) + m_bias2(col) + m_min;
            }
            else {
                return ublas::inner_prod(matrix_row(m_data1, row), matrix_row(m_data1, col)) + m_bias1(row) + m_bias1(col) + m_min;
            }
        }

        float loop (std::vector<Entry> const &data) {
            matrix &col_data = m_size2 ? m_data2 : m_data1;
            vector &col_bias = m_size2 ? m_bias2 : m_bias1;
            matrix &col_delta = m_size2 ? m_delta2 : m_delta1;
            vector &col_bias_delta = m_size2 ? m_bias_delta2 : m_bias_delta1;
            float err = 0;
            progress_display progress(data.size(), cerr);
            for (Entry const &e: data) {
                matrix_row r1(m_data1, e.row);
                matrix_row r2(col_data, e.col);
                matrix_row d1(m_delta1, e.row);
                matrix_row d2(col_delta, e.col);
                float &b1 = m_bias1(e.row);
                float &b2 = col_bias(e.col);
                float &bd1 = m_bias_delta1(e.row);
                float &bd2 = col_bias_delta(e.col);

                if (m_options.m_mom == 0) {
                    d1 = d2 = ublas::scalar_vector<float>(m_options.m_dim, 0);
                    bd1 = bd2 = 0;
                }
                else {
                    d1 *= m_options.m_mom;
                    d2 *= m_options.m_mom;
                    bd1 *= m_options.m_mom;
                    bd2 *= m_options.m_mom;
                }
                float predict = inner_prod(r1, r2) + b1 + b2 + m_min;
                float delta = e.value - predict;
                err += delta * delta;
                d1 += -delta * r2 + m_options.m_r1 * r1;
                d2 += -delta * r1 + m_options.m_r2 * r2;
                bd1 += -delta;
                bd2 += -delta;
                r1 -= m_options.m_eps * d1;
                r2 -= m_options.m_eps * d2;
                b1 -= m_options.m_eps * bd1;
                b2 -= m_options.m_eps * bd2;
                ++progress;
            }
            return sqrt(err / data.size());
        }
    };
}



#endif

