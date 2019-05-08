#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/args.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/module.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/overloads.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/tuple.hpp>
#pragma GCC diagnostic pop

#include <iostream>
#include <map>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>

#include <time.h>

typedef std::vector<std::string> StringVec;
typedef std::vector<std::vector<double> > DoubleMatrix;

namespace py = boost::python;

StringVec str_vector_from_python_list (const boost::python::list &input_list)
{
    int n_str = boost::python::len(input_list);
    StringVec output (0);

    for (int i = 0; i < n_str; i++)
        output.push_back(boost::python::extract<std::string>(input_list[i]));

    return output;
}

boost::python::list list_from_str_vector (StringVec &input_vec)
{
    boost::python::list out_list;

    for (int i = 0; i < input_vec.size (); ++i)
        out_list.append (input_vec[i]);

    return out_list;
}

DoubleMatrix boost_multi_array_2d_from_numpy_array (
    const boost::python::object &np_arr)
{
    boost::python::object size_tuple = boost::python::extract<boost::python::tuple> (np_arr.attr("shape"));
    int n_rows = boost::python::extract<int>(size_tuple[0]);
    int n_cols = boost::python::extract<int>(size_tuple[1]);

    DoubleMatrix bma (n_rows, std::vector<double>(n_cols));

    for (int i = 0; i < n_rows; ++i)
        for (int j = 0; j < n_cols; ++j)
            bma[i][j] = boost::python::extract<double>(np_arr[i][j]);

    return bma;
}

boost::python::list nested_list_from_boost_multi_array_2d (const DoubleMatrix &bma)
{
    int n_rows = bma.size ();
    int n_cols = bma[0].size ();
    boost::python::list l;

    for (int i = 0; i < n_rows; i++)
    {
        boost::python::list ll;

        for (int j = 0; j < n_cols; j++)
            ll.append (bma[i][j]);
        l.append (ll);
    }

    return l;
}

boost::python::list merge_timeframes (const boost::python::object &fast_feat_arg,
                                        const boost::python::object &slow_feat_arg,
                                        const boost::python::list &fast_idx_arg,
                                        const boost::python::list &slow_idx_arg,
                                        const int &delay)
{
    DoubleMatrix fast_feat = boost_multi_array_2d_from_numpy_array (fast_feat_arg);
    DoubleMatrix slow_feat = boost_multi_array_2d_from_numpy_array (slow_feat_arg);

    StringVec fast_idx = str_vector_from_python_list (fast_idx_arg);
    StringVec slow_idx = str_vector_from_python_list (slow_idx_arg);

    DoubleMatrix output (fast_feat.size(), std::vector<double>(fast_feat[0].size() + slow_feat[0].size()));

    int idx_slow_frame, idx_fast_frame;

    idx_slow_frame = 0;

    for (int i = 0; i < fast_idx.size(); i++)
    {
        struct tm tm;
        strptime(fast_idx[i].c_str(), "%Y-%m-%d %H:%M:%S", &tm);
        time_t ts1 = mktime(&tm) - delay * 60 * 60 * 24;  // t is now your desired time_t
        time_t ts2 = ts1 - 10 * 24 * 60 * 60;
        char buffer1[20];
        char buffer2[20];

        strftime(buffer1, 20, "%Y-%m-%d %H:%M:%S", localtime(&ts1));
        strftime(buffer2, 20, "%Y-%m-%d %H:%M:%S", localtime(&ts2));

        std::string t_stamp1 (buffer1);
        std::string t_stamp2 (buffer2);
        //row_h = ds_h.f_df.loc[ts2:ts].ix[-1,:]
        //finds corresponding row in the slow timeframe
        for (; slow_idx[idx_slow_frame] < t_stamp1 && idx_slow_frame < slow_idx.size (); idx_slow_frame++)
        {
        }

        //first copies the features from the fast timeframe
        for (int j = 0; j < fast_feat[0].size(); j++)
            output [i] [j] = fast_feat[i][j];
        //then, copies the features from the slow one
        for (int j = 0; j < slow_feat[0].size(); j++)
            output [i] [j + fast_feat[0].size()] = slow_feat[i][j];
        //new_feat_arr[i, :] = row_h

    }

    return nested_list_from_boost_multi_array_2d (output);
}

BOOST_PYTHON_MODULE(cpp_utils)
{
    namespace py = boost::python;

    py::class_<StringVec>("StringVec")
        .def(py::vector_indexing_suite<StringVec>());

    py::class_<DoubleMatrix>("DoubleMatrix");

    py::def("boost_multi_array_2d_from_numpy_array",
        boost_multi_array_2d_from_numpy_array, py::args("bma"),
        "Converts 2d np.array to boost::multi_array");

    py::def("nested_list_from_boost_multi_array_2d",
        nested_list_from_boost_multi_array_2d, py::args("bma"),
        "Converts boost::multi_array to 2d np.array");

    py::def("list_from_str_vector",
            list_from_str_vector, py::args ("str_v"),
            "Tests implementation");

    py::def("str_vector_from_python_list",
            str_vector_from_python_list, py::args ("l"),
            "Tests implementation");

    py::def("merge_timeframes", merge_timeframes,
            py::args("fast_df", "slow_df", "fast_idx", "slow_idx", "delay"),
            "Merges two timeframes");

}


//compilation
//g++ -I /usr/include/python2.7 -fpic -c -o cpp_utils.o cpp_utils.cpp
//g++ -o cpp_utils.so -shared cpp_utils.o -lboost_python -lpython2.7
