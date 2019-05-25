/**
 *  Author: Rohan Ramasamy
 *  Date: 25/05/2019
 */

#include <vector>
#include <utility>

namespace mcf 
{
	class Interpolator1D {
	public:
		/**
		 * Constructor storing data for x and y values
		 */
		Interpolator1D(
			std::vector<double> xValues,
			std::vector<double> yValues
			);

		/**
		 * Find the corresponding y for x
		 */
		double
		interpY(
			const double& interpolatedX
			) const;

	private:
		/**
		 * Function used to get the index and scaling between the two closest points
		 */
		std::pair<int, double>
		getIndexAndScaling(
			const double& interpolatedX
			) const;

		// Data of x and y values to interpolate
		std::vector<double> mXValues, mYValues;
	};

}
