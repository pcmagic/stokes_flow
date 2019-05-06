#include <boost/math/special_functions/bessel.hpp>
#include <iostream>
#include <iterator>
#include <algorithm>

int main()
{
    using namespace boost::math;
    std::cout << cyl_bessel_i(1000, 10) << " " << std::endl;
}