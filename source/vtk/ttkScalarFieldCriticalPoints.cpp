#include <Gale.h>
#include <MemoryUsage.h>
#include <ScalarFieldCriticalPoints.h>
#include <vtkDataSet.h>
#include <vtkSignedCharArray.h>
#include <vtkXMLGenericDataObjectReader.h>
#include <iostream>
#include <unistd.h>

#include "ttkUtils.h"

bool VertexIds{true}, VertexScalars{true}, VertexBoundary{false};

int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " -i <input.vtu>" << std::endl;
        std::cout << "Missing required input .vtu file!" << std::endl;
        return EXIT_FAILURE;
    }

    int c;
    int blockNumber = 0;
    int blockSize = 0;
    int threadNumber = 4;
    int numThreadsPerCluster = 1;
    std::string fileName = "";
    std::string inputArrayName = "";

    while ((c = getopt(argc, argv, "a:b:i:p:s:t:")) != -1) {
        switch (c) {
            case 'i':
                fileName = optarg;
                break;
            case 'a':
                inputArrayName = optarg;
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
    vtkSmartPointer<vtkXMLGenericDataObjectReader> reader =
        vtkSmartPointer<vtkXMLGenericDataObjectReader>::New();
    reader->SetFileName(fileName.c_str());
    reader->Update();
    
    MemoryUsage m;

    int scalarFieldIndex = 0;
    vtkDataObject *inputDataObject = reader->GetOutput();
    if(!inputDataObject) {
        std::cerr << "Unable to read input file `" << fileName << "' :(" << std::endl;
        return 1;
    }
    vtkDataSet *inputDataSet = vtkDataSet::SafeDownCast(inputDataObject);
    // get the scalar field array to process 
    GetInputArray(inputArrayName, inputDataSet, scalarFieldIndex);

    ttk::Gale gale(threadNumber, numThreadsPerCluster);
    if (blockNumber > 0 && blockSize > 0)
        gale.initialize(inputDataSet, blockNumber, blockSize, false);
    else if (blockNumber > 0)
        gale.initialize(inputDataSet, blockNumber);
    else 
        gale.initialize(inputDataSet);

    if(VertexBoundary)
        gale.preconditionBoundaryVertices();

    std::vector<std::pair<ttk::SimplexId, char>> criticalPoints_;

    ttk::ScalarFieldCriticalPoints cpPlugin;
    cpPlugin.preconditionTriangulation(&gale);
    cpPlugin.setThreadNumber(threadNumber);
    cpPlugin.setOutput(&criticalPoints_);
    
    vtkDataArray *offsetField = GetOrderArray(inputDataSet, scalarFieldIndex, 1);

    ttk::Timer t;

    int res = cpPlugin.executeLegacy(static_cast<ttk::SimplexId *>(GetVoidPointer(offsetField)), &gale);
    if (res != 0) {
        std::cerr << "Execution failed, status code: " << res << std::endl;
    }

    if(VertexBoundary) {
        vtkNew<vtkSignedCharArray> vertexBoundary{};
        vertexBoundary->SetNumberOfComponents(1);
        vertexBoundary->SetNumberOfTuples(criticalPoints_.size());
        vertexBoundary->SetName("IsOnBoundary");

    #ifdef TTK_ENABLE_OPENMP
    #pragma omp parallel for num_threads(threadNumber)
    #endif
        for(size_t i = 0; i < criticalPoints_.size(); i++) {
        vertexBoundary->SetTuple1(
            i, (signed char)gale.isVertexOnBoundary(
                criticalPoints_[i].first));
        }
    }

    std::cout << "[ScalarFieldCriticalPoints] Time usage for computation: " << t.getElapsedTime() << " s." << std::endl;
    std::cout << "[ScalarFieldCriticalPoints] Memory usage: " << m.getValue_in_MB(false) << " MB." << std::endl;

    return 0;
}
