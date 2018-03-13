"""
Author: Rohan Ramasamy
Data: 27/10/2017

This file contains vtk writers to save results to file
"""

import numpy as np
import vtk


def write_vti_file(output_array, main_name):
    """
    Write a simple vti file, given a 3D array
    """
    output_file_name = "{}.vti".format(main_name)
    dims = output_array.shape
    assert len(dims) == 3

    imageData = vtk.vtkImageData()
    imageData.SetDimensions(dims[0], dims[1], dims[2])
    imageData.AllocateScalars(vtk.VTK_DOUBLE, 1)

    # Fill every entry of the image data with "2.0"
    for z in range(dims[2]):
        for y in range(dims[1]):
            for x in range(dims[0]):
                imageData.SetScalarComponentFromDouble(x, y, z, 0, output_array[x, y, z])

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_file_name)
    writer.SetInputData(imageData)
    writer.Write()


if __name__ == '__main__':
    output = np.random.random((10, 11, 12))
    file_name = "test_output"

    write_vti_file(output, file_name)

