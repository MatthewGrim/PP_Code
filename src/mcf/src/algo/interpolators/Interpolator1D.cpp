/**
 *  Author: Rohan Ramasamy
 *  Date: 25/05/2019
 */

 #include <mcf/src/algo/interpolators/Interpolator1D.h>
 
 #include <cassert>
 #include <math.h>
 #include <stdexcept>
 #include <iostream>
 #include <sstream>

 namespace mcf 
 {
 	Interpolator1D::
	Interpolator1D(
		std::vector<double> xValues,
		std::vector<double> yValues
		) :
	    mXValues(xValues),
	    mYValues(yValues)
    {
    	// Run checks on data
    	if (xValues.size() != yValues.size()) throw std::runtime_error("Inconsistent data set sizes!");
    	if (xValues.size() < 3) throw std::runtime_error("Data set is too small!");

    	for (size_t i = 0; i < xValues.size() - 1; ++i) {
    		if (xValues[i] > xValues[i + 1]) throw std::runtime_error("x values are not monotonic!");
    	}
    }

	std::pair<int, double>
	Interpolator1D::
	getIndexAndScaling(
		const double& interpolatedX
		) const
	{
		// Check interpolated alpha is within the range of data
		if (mXValues[0] > interpolatedX || mXValues[mXValues.size() - 1] < interpolatedX) {
			std::stringstream ss;
			ss << "x: " << interpolatedX << " is outside of range!";
			throw std::runtime_error(ss.str());
		} 
		if (isnan(interpolatedX)) throw std::runtime_error("x is NaN!");

		// Find closest index below interpolated value by bisection
		size_t idx = mXValues.size() / 2;
		size_t lowIdx = 0;
		size_t highIdx = mXValues.size() - 1;
		bool foundIdx = false;
		while (!foundIdx) {
			if (mXValues[idx] < interpolatedX) {
				lowIdx = idx;
				idx = (highIdx + lowIdx) / 2;
			}
			else if (mXValues[idx] == interpolatedX) {
				// Do nothing
			}
			else {
				highIdx = idx;
				idx = (highIdx + lowIdx) / 2;
			}

			foundIdx = (mXValues[idx] <= interpolatedX) && 
			           (mXValues[idx + 1] >= interpolatedX);
		} 

		// Get scaling
		double scaling = (interpolatedX - mXValues[idx]) / (mXValues[idx + 1] - mXValues[idx]);

		return std::make_pair(idx, scaling);
	}

    double
    Interpolator1D::
	interpY(
		const double& interpolatedX
		) const
	{
		auto idxAndScaling = getIndexAndScaling(interpolatedX);

		double lowerY = mYValues[idxAndScaling.first];
		double upperY = mYValues[idxAndScaling.first + 1];

		return lowerY + idxAndScaling.second * (upperY - lowerY);
	}

 }
