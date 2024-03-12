#include <Gale.h>
#include <MemoryUsage.h>
#include <FTMTree.h>
#include <vtkAlgorithm.h>
#include <vtkConnectivityFilter.h>
#include <vtkSetGet.h>
#include <vtkThreshold.h>
#include <vtkXMLGenericDataObjectReader.h>
#include <vtkUnstructuredGrid.h>
#include <iostream>
#include <unistd.h>

#include "ttkUtils.h"
#include "ttkFTMStructures.h"

// Gale parameters
int blockNumber_ = 32;
int blockSize_ = 400;
int threadNumber_ = 1;
int numThreadsPerCluster_ = 40;
std::string inputArrayName_ = "";

// FTM tree parameters
int nbCC_{};
std::vector<int> arrayIndices_;
ttk::ftm::Params params_;
std::vector<vtkSmartPointer<vtkDataSet>> connected_components_;
std::vector<ttk::Gale *> triangulation_;
std::vector<ttk::ftm::LocalFTM> ftmTree_;
std::vector<vtkDataArray *> inputScalars_;
std::vector<std::vector<ttk::SimplexId>> offsets_;


void identify(vtkDataSet *ds) {
    vtkNew<ttkSimplexIdTypeArray> identifiers{};
    const auto nbPoints = ds->GetNumberOfPoints();
    identifiers->SetName(ttk::VertexScalarFieldName);
    identifiers->SetNumberOfComponents(1);
    identifiers->SetNumberOfTuples(nbPoints);

    for(int i = 0; i < nbPoints; i++) {
        identifiers->SetTuple1(i, i);
    }

    ds->GetPointData()->AddArray(identifiers);
}

int preconditionTriangulation() {
    triangulation_.resize(nbCC_);
    ftmTree_.resize(nbCC_);
    
    if (nbCC_ == 1) {
        triangulation_[0]
            = new ttk::Gale(threadNumber_, numThreadsPerCluster_);

        triangulation_[0]->initialize(connected_components_[0], blockNumber_, blockSize_, false);

        ftmTree_[0].tree.setThreadNumber(threadNumber_);
        ftmTree_[0].tree.preconditionTriangulation(triangulation_[0]);
    }
    else {
        for(int cc = 0; cc < nbCC_; cc++) {
            triangulation_[cc]
                = new ttk::Gale(threadNumber_, numThreadsPerCluster_);

            triangulation_[cc]->initialize(connected_components_[cc], blockNumber_, blockSize_, true);

            ftmTree_[cc].tree.setThreadNumber(threadNumber_);
            ftmTree_[cc].tree.preconditionTriangulation(triangulation_[cc]);

            // #ifndef TTK_ENABLE_KAMIKAZE
            //     if(triangulation_[cc]->isEmpty()) {
            //     this->printMsg({"Error : ttkTriangulation on connected component",
            //                     std::to_string(cc), " allocation problem."},
            //                     ttk::debug::Priority::ERROR);
            //     return 0;
            //     }
            // #endif
        }
    }

    return 1;
}

int setOrderArrays() {
    arrayIndices_.resize(nbCC_);
    for(int cc = 0; cc < nbCC_; cc++) {
        GetInputArray(inputArrayName_, connected_components_[cc], arrayIndices_[cc]);
        GetOrderArray(connected_components_[cc], arrayIndices_[cc], 0);
    }
    return 1;
}

int getScalars() {
    inputScalars_.resize(nbCC_);
    for(int cc = 0; cc < nbCC_; cc++) {
        inputScalars_[cc] = connected_components_[cc]->GetPointData()->GetArray(arrayIndices_[cc]);

        if(!inputScalars_[cc]) {
            std::cerr << "Error : input scalar " << cc << " field pointer is null." << std::endl;
            return 0;
        }
    }

    return 1;
}

int getOffsets() {
    // should be called after getScalars for inputScalars_ needs to be filled
    offsets_.resize(nbCC_);
    for(int cc = 0; cc < nbCC_; cc++) {
        const auto offsets = GetOrderArray(connected_components_[cc], arrayIndices_[cc], 0);
        offsets_[cc].resize(connected_components_[cc]->GetNumberOfPoints());

        for(size_t i = 0; i < offsets_[cc].size(); i++) {
            offsets_[cc][i] = offsets->GetTuple1(i);
        }

        if(offsets_[cc].empty()) {
            std::cerr << "Error : wrong input offset scalar field for " << cc << std::endl;
            return 0;
        }
    }

    return 1;
}

ttk::ftm::TreeType GetTreeType(void){
    return params_.treeType;
}

void SetTreeType(const int type) {
    params_.treeType = (ttk::ftm::TreeType)type;
}

bool GetWithSegmentation(void) {
    return params_.segm;
}

bool GetWithNormalize(void) {
    return params_.normalize;
}


int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " -i <input.vtu>" << std::endl;
        std::cout << "Missing required input .vtu file!" << std::endl;
        return EXIT_FAILURE;
    }

    int c;
    int treeType{};
    std::string fileName = "";

    while ((c = getopt(argc, argv, "a:b:i:p:s:t:T:")) != -1) {
        switch (c) {
            case 'i':
                fileName = optarg;
                break;
            case 'a':
                inputArrayName_ = optarg;
                break;
            case 'b':
                blockNumber_ = atoi(optarg);
                break;
            case 'p':
                numThreadsPerCluster_ = atoi(optarg);
                break;
            case 's':
                blockSize_ = atoi(optarg);
                break;
            case 't':
                threadNumber_ = atoi(optarg);
                break;
            case 'T':
                treeType = atoi(optarg);
                break;
        }
    }

    std::cout << "Parsing vtu file: " << fileName << std::endl;
    
    //read all the data from the file
    vtkSmartPointer<vtkXMLGenericDataObjectReader> reader =
        vtkSmartPointer<vtkXMLGenericDataObjectReader>::New();
    reader->SetFileName(fileName.c_str());
    reader->Update();
    

    vtkDataObject *inputDataObject = reader->GetOutput();
    if(!inputDataObject) {
        std::cerr << "Unable to read input file `" << fileName << "' :(" << std::endl;
        return 1;
    }
    vtkDataSet *inputDataSet = vtkDataSet::SafeDownCast(inputDataObject);
    SetTreeType(treeType);

    MemoryUsage m;
    
    // This data set may have several connected components,
    // we need to apply the FTM Tree for each one of these components
    // We then reconstruct the global tree using an offest mecanism
    identify(inputDataSet);

    // vtkNew<vtkConnectivityFilter> connectivity{};
    // connectivity->SetInputData(inputDataSet);
    // connectivity->SetExtractionModeToAllRegions();
    // connectivity->ColorRegionsOn();
    // connectivity->Update();

    // nbCC_ = connectivity->GetOutput()
    //           ->GetCellData()
    //           ->GetArray("RegionId")
    //           ->GetRange()[1]
    //         + 1;
    nbCC_ = 1;      // force the contour tree computation on the entire dataset
    connected_components_.resize(nbCC_);

    // if(nbCC_ > 1) {
    //   // Warning, in case of several connected components, the ids seen by
    //   // the base code will not be consistent with those of the original
    //   // mesh
    //   for(int cc = 0; cc < nbCC_; cc++) {
    //     vtkNew<vtkThreshold> threshold{};
    //     threshold->SetInputConnection(connectivity->GetOutputPort());
    //     threshold->SetInputArrayToProcess(
    //       0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_CELLS, "RegionId");
    //     threshold->ThresholdBetween(cc, cc);
    //     threshold->Update();
    //     connected_components_[cc] = vtkSmartPointer<vtkUnstructuredGrid>::New();
    //     connected_components_[cc]->ShallowCopy(threshold->GetOutput());
    //   }
    // } else {
        connected_components_[0] = vtkSmartPointer<vtkUnstructuredGrid>::New();
        connected_components_[0]->ShallowCopy(inputDataSet);
    // }

    setOrderArrays();
    // std::cout << "[FTMTree] Memory usage before preconditioning triangulation: " << m.getValue_in_MB(false) << " MB." << std::endl;

    // now proceed for each triangulation obtained.
    if(preconditionTriangulation() == 0) {
        std::cerr << "Error : wrong triangulation." << std::endl;
        return 0;
    }
    // std::cout << "[FTMTree] Memory usage after preconditioning triangulation: " << m.getValue_in_MB(false) << " MB." << std::endl;

    // Fill the vector of scalar/offset, cut the array in pieces if needed
    if(getScalars() == 0) {
        std::cerr << "Error : wrong input scalars." << std::endl;
        return 0;
    }
    getOffsets();

    std::cout << "Launching on field: " << inputScalars_[0]->GetName() << std::endl;

    ttk::ftm::idNode acc_nbNodes = 0;
    

    // Build tree
    for(int cc = 0; cc < nbCC_; cc++) {
        ftmTree_[cc].tree.setVertexScalars(
            GetVoidPointer(inputScalars_[cc]));
        ftmTree_[cc].tree.setVertexSoSoffsets(offsets_[cc].data());
        ftmTree_[cc].tree.setTreeType(GetTreeType());
        ftmTree_[cc].tree.setSegmentation(GetWithSegmentation());
        ftmTree_[cc].tree.setNormalizeIds(GetWithNormalize());

        switch(inputScalars_[cc]->GetDataType()) {
            vtkTemplateMacro(ftmTree_[cc].tree.build<VTK_TT>(triangulation_[cc]));
        }

        ftmTree_[cc].offset = acc_nbNodes;
        acc_nbNodes += ftmTree_[cc].tree.getTree(GetTreeType())->getNumberOfNodes();
    }
    std::cout << "[ttkFTMTree] number of nodes: " << acc_nbNodes << std::endl;

    // // Construct output
    // if(getSkeletonNodes(outputSkeletonNodes) == 0) {
    //     std::cerr << "Error : wrong properties on skeleton nodes." << std::endl;
    //     return 0;
    // }

    // if(getSkeletonArcs(outputSkeletonArcs) == 0) {
    //     std::cerr << "Error : wrong properties on skeleton arcs." << std::endl;
    //     return 0;
    // }

    // if(GetWithSegmentation()) {
    //     outputSegmentation->ShallowCopy(input);
    //     if(getSegmentation(outputSegmentation) == 0) {
    //         std::cerr << "Error : wrong properties on segmentation." << std::endl;
    //         return 0;
    //     }
    // }

    std::cout << "[FTMTree] Memory usage: " << m.getValue_in_MB(false) << " MB." << std::endl;

    return 0;
}
