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
            unsigned m_dim;
            float m_r1, m_r2;   // regulation terms
            float m_mom, m_eps, m_init;
            float m_min;
            float m_th;
            unsigned m_maxit;
            friend class Embed;
        public:
            void add (po::options_description_easy_init init, string const &prefix = "embed") {
                init
                    ((prefix + "-dim").c_str(), po::value(&m_dim)->default_value(20), "dimension")
                    ((prefix + "-r1").c_str(), po::value(&m_r1)->default_value(0.1), "regulate")
                    ((prefix + "-r2").c_str(), po::value(&m_r2)->default_value(0.1), "regulate")
                    ((prefix + "-mom").c_str(), po::value(&m_mom)->default_value(0.9), "moment")
                    ((prefix + "-eps").c_str(), po::value(&m_eps)->default_value(0.01), "")
                    ((prefix + "-init").c_str(), po::value(&m_init)->default_value(0.1), "")
                    ((prefix + "-min").c_str(), po::value(&m_min)->default_value(nanf("")), "")
                    ((prefix + "-th").c_str(), po::value(&m_th)->default_value(0.5), "")
                    ((prefix + "-maxit").c_str(), po::value(&m_maxit)->default_value(0), "")
                    //(prefix + "-max", po::value(&m_max)->default_value(nanf("")), "")
                ;
            }
        };
    private:
        unsigned m_size1, m_size2;
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
    public:
        struct Entry {
            unsigned row, col;
            float value;
        };

        Embed (Options const &opt): m_options(opt) {
        }

        void save () const {
        }

        void load () {
        }

        void init (unsigned s1, unsigned s2, std::vector<Entry> const &data) {
            m_size1 = s2;
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
        float predict (unsigned row, unsigned col) const {
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

        /*
        void train (vector<Entry> const &data) {
        }
        */
        void train (unsigned s1, unsigned s2, std::vector<Entry> const &data) {
            init(s1, s2, data);
            for (unsigned it = 0; m_options.m_maxit == 0 || it < m_options.m_maxit; ++it) {
                float v = loop(data);
                cerr << it << '\t' << v << endl;
                if (v < m_options.m_th) break;
            }
        }
    };
}



#endif

