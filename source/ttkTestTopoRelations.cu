#include "ttk/AcTopo.h"
#include "ttk/ExplicitTriangulation.h"
#include "ttk/TestTopoRelations.h"
#include <vtkSmartPointer.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkUnstructuredGrid.h>
#include <iostream>
#include <stdio.h>
#include <unistd.h>


int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " -i <input.vtu>" << std::endl;
        std::cout << "Missing required input .vtu file!" << std::endl;
        return EXIT_FAILURE;
    }

    int c;
    std::string fileName = "";
    int blockNumber = 0;
    int blockSize = 0;
    int threadNumber = 1;
    int numThreadsPerCluster = 1;

    while ((c = getopt(argc, argv, "a:b:i:p:s:t:")) != -1) {
        switch (c) {
            case 'i':
                fileName = optarg;
                break;
            case 'b':
                blockNumber = atoi(optarg);
                break;
            case 'p':
                numThreadsPerCluster = atoi(optarg);
                break;
            case 's':
                blockSize = atoi(optarg);
                break;
            case 't':
                threadNumber = atoi(optarg);
                break;
        }
    }

    std::cout << "Parsing vtu file: " << fileName << std::endl;
    
    //read all the data from the file
    vtkSmartPointer<vtkXMLUnstructuredGridReader> reader =
        vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
    reader->SetFileName(fileName.c_str());
    reader->Update();

    // ttk::ExplicitTriangulation explicitTriangulation;
    // explicitTriangulation.createExplicitTriangulation(reader->GetOutput());

    // ttk::TestTopoRelations testPlugin;
    // testPlugin.preconditionTriangulation(&explicitTriangulation);
    // testPlugin.execute(&explicitTriangulation);

    ttk::AcTopo actopo(threadNumber, numThreadsPerCluster);
    if (blockNumber > 0 && blockSize > 0)
        actopo.initialize(reader->GetOutput(), blockNumber, blockSize);
    else if (blockNumber > 0)
        actopo.initialize(reader->GetOutput(), blockNumber);
    else 
        actopo.initialize(reader->GetOutput());

    ttk::TestTopoRelations testPlugin;
    testPlugin.preconditionTriangulation(&actopo);
    testPlugin.setThreadNumber(threadNumber);
    testPlugin.execute(&actopo);

    return 0;
}
