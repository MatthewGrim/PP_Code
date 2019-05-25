/*
Author: Rohan Ramasamy
Data: 24/05/2019

This file contains unit tests for Interpolator1D
*/

#include <mcf/src/algo/interpolators/Interpolator1D.h>

#include <gtest/gtest.h>

#include <vector>
#include <iostream>


namespace mcf 
{
	class Interpolator1DTest : public ::testing::Test {
	public:
		virtual void SetUp() {}

		virtual void TearDown() {}
	};

	TEST_F(Interpolator1DTest, ConstructorTest) {
		// Mismatched sizes
		std::vector<double> alphas = {0.0, 1.0, 2.0, 3.0};
		std::vector<double> liftCoefficients = {0.0, 1.0};
		EXPECT_ANY_THROW(Interpolator1D(alphas, liftCoefficients));

		// Non-monotonic alpha
		alphas = {1.0, 0.5, 2.0};
		liftCoefficients = {0.0, 1.0, 2.0, 3.0};
		EXPECT_ANY_THROW(Interpolator1D(alphas, liftCoefficients));

		// Data set too small
		alphas = {1.0, 0.5};
		liftCoefficients = {0.0, 1.0};
		EXPECT_ANY_THROW(Interpolator1D(alphas, liftCoefficients));
	}

	TEST_F(Interpolator1DTest, InterpolateOnDataPoints) {
		std::vector<double> alphas = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
		std::vector<double> liftCoefficients = { 0.5, 0.575, 0.65, 0.725, 0.8, 0.875, 0.95, 1.025, 1.1, 1.1, 1.0 };
		Interpolator1D interpolator(alphas, liftCoefficients);

		// Check interpolated data set values are exact
		for (size_t i = 0; i < alphas.size(); ++i) {
			EXPECT_EQ(liftCoefficients[i], interpolator.interpY(alphas[i]));
		}
	}

	TEST_F(Interpolator1DTest, InterpolateOutsideDataLimits) {
		std::vector<double> alphas = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
		std::vector<double> liftCoefficients = { 0.5, 0.575, 0.65, 0.725, 0.8, 0.875, 0.95, 1.025, 1.1, 1.1, 1.0 };
		Interpolator1D interpolator(alphas, liftCoefficients);

		// Check error thrown when bounds exceeded
		EXPECT_ANY_THROW(interpolator.interpY(10.6));
		EXPECT_ANY_THROW(interpolator.interpY(-0.1));
	}
}
