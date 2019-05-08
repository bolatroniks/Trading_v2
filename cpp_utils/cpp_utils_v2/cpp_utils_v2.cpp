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
#include <sstream>
#include <map>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>

#include <time.h>
#include <iostream>
#include <fstream>

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
    std::ofstream myfile;
    myfile.open ("debug.txt", std::ios_base::app);

    myfile << "Test\n";
    myfile.close ();

    boost::python::object size_tuple = boost::python::extract<boost::python::tuple> (np_arr.attr("shape"));
    int n_rows = boost::python::extract<int>(size_tuple[0]);
    int n_cols = boost::python::extract<int>(size_tuple[1]);
    char factor[5];


    DoubleMatrix bma (n_rows, std::vector<double>(n_cols));

    std::cout << "Test";

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

DoubleMatrix merge_frames (const StringVec &fast_idx,
                                        const StringVec &slow_idx,
                                        //const DoubleMatrix &fast_feat,
                                        const DoubleMatrix &slow_feat,
                                        const int &offset)
{
    std::vector<int> fast_to_slow_indexing (fast_idx.size (), 0);
    int i = 0;

    for (int j = 0; j < slow_idx.size () - 1; j++)
    {
        while ((fast_idx[i] >= slow_idx[j]) &&
         (fast_idx[i] < slow_idx[j+1]) &&
            (i < fast_idx.size() - 1))
        {
                fast_to_slow_indexing [i] = j - offset;
                i++;
        }
    }

    //-----------------------------------------------------#
    //--------implements the delay-------------------------#
    int burned = 0;
    while (fast_to_slow_indexing[burned] < 0) burned++;

    DoubleMatrix new_feat_arr (fast_idx.size() - burned,
                        std::vector<double>(//fast_feat[0].size() +
                                            slow_feat[0].size()));

    //for (int i = 0; i < new_feat_arr.size (); i++)
    //    for (int j = 0; j < fast_feat[0].size(); j++)
    //        new_feat_arr[i] [j] = fast_feat [i + burned] [j];

    for (int i = 0; i < new_feat_arr.size (); i++)
        for (int j = 0;  j < slow_feat[0].size (); j++)
            new_feat_arr[i] [j /*+ fast_feat[0].size()*/] = slow_feat [fast_to_slow_indexing [i + burned]] [j];

    return new_feat_arr;
}

BOOST_PYTHON_MODULE(cpp_utils_v2)
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

    py::def("merge_frames", merge_frames,
            py::args("fast_idx", "slow_idx", "fast_arr", /*"slow_arr",*/ "delay"),
            "Merges two timeframes");

}


//compilation
//g++ -I /usr/include/python2.7 -fpic -c -o cpp_utils.o cpp_utils.cpp
//g++ -o cpp_utils.so -shared cpp_utils.o -lboost_python -lpython2.7

// edit /etc/sysctl.d/10-ptrace.conf
// echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
