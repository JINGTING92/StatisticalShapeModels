/*=========================================================================

Program:   Medical Imaging & Interaction Toolkit
Language:  C++
Date:      $Date$
Version:   $Revision$ 
 
Copyright (c) German Cancer Research Center, Division of Medical and
Biological Informatics. All rights reserved.
See MITKCopyright.txt or http://www.mitk.org/copyright.html for details.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#ifndef NonlinearModelling_h
#define NonlinearModelling_h

#include <berryISelectionListener.h>

#include <QmitkAbstractView.h>

#include "ui_NonlinearModellingControls.h"

#include <sstream>

// Qt
#include <QMessageBox>
#include <qfiledialog.h>
#include <qslider.h>
#include <qSpinBox.h>
#include <qcombobox.h>
#include <qlistview.h>
#include <qtreeview.h>

// Mitk
#include <mitkDataNodeFactory.h>
#include "mitkNodePredicateDimension.h"
#include "mitkNodePredicateDataType.h"
#include "mitkNodePredicateAnd.h"
#include <mitkNodePredicateProperty.h>
#include <mitkSurface.h>
#include <mitkImageCast.h>
#include <mitkImage.h>
#include <mitkLookupTable.h>
#include <mitkLookupTableProperty.h>
#include <mitkMaterial.h>
#include <mitkProperties.h>

#include "mitkImageToSurfaceFilter.h"

// Itk
#include <itkImage.h>
#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageDuplicator.h>
#include <itkImageToVTKImageFilter.h>
#include <itkVTKImageExport.h>
#include <itkRegionOfInterestImageFilter.h> 

// Vtk
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
#include <vtkAppendPolyData.h>
#include <vtkCleanPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkProcrustesAlignmentFilter.h>
#include <vtkLandmarkTransform.h>
#include <vtkDecimatePro.h>
#include <vtkMarchingCubes.h>
#include <vtkImageData.h>
#include <vtkDoubleArray.h>
#include <vtkMath.h> 
#include <vtkDataSetAttributes.h>
#include <vtkPointData.h>
#include <vtkCenterOfMass.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkIdTypeArray.h>
#include <vtkSelectionNode.h>
#include <vtkInformation.h>
#include <vtkSelection.h>
#include <vtkExtractSelection.h>
#include <vtkSphereSource.h>
#include <vtkUnstructuredGrid.h>
#include <vtkGeometryFilter.h>
#include <vtkPointSource.h> 
#include <vtkIterativeClosestPointTransform.h>
#include <vtkPolyDataNormals.h> 

// internal calls 
#include "RobustKPCA.h"


class NonlinearModelling : public QmitkAbstractView
{
    Q_OBJECT
public:  

    static const std::string VIEW_ID;

    virtual void CreateQtPartControl(QWidget *parent);

	NonlinearModelling(); 

	~NonlinearModelling();
	
protected slots:
    
     // =================================== Model Training ============================================ //
     

     void LoadTrainingData(); 

	 void GenerateIncompleteDatasets(); 

	 void GenerateLousyDatasets(); 

	 void ResetDatasets(); 

	 void KernelPCA(); 

	 void RobustPCA(); 

	 void Miccai17(); 

	 void RKPCA(); 

	 // ====================================== Evaluation =============================================== //
	 

	 /** 
	  * @brief  Random sampling model pdf and compute Generalization ability & Speciticity 
	  *         Generalization ability : Compute the nearest sample with training datasets 
	  *         Specificity : Compute the nearest input sample 
	  * @param  choose number of samples + order of mode 
	  *         Set the number of each mode regarding the length of this mode , utilize the proportion 
	  */
	 void EvaluationGS(); 

	 /**
	  * @brief  Back project GT onto model and get the reconstructions
	  *         Compute the reconstruction with input 
	  */
	 void EvaluationDistGT();


	 /*******************************/


	 /**
	  * @brief  Load GT for evaluation of corrupted training datasets   
	  */
	 void EvaluationLoadGTForCorruption(); 


	 void LoadTestGTForBackProjection(); 


	 /**
	  * @brief  Load test datasets and project onto the model 
	  *         Comppute the distance with input datasets with projected 
	  */
	 void LoadTestForBackProjection();


	 // ================================= Additional Tools =========================================== // 

	 /**
	  *  @brief   Load training polys to estimate the gamma
	  *           Find mean the nearest neighborhood distance of every training sample
	  */
	 void SigmaEstimation();


protected:

    virtual void SetFocus();

	void AlignDataSetsWithScaling(std::vector< vtkSmartPointer< vtkPolyData >> &m_meshes);

	void AlignDataSetsWithoutScaling(std::vector< vtkSmartPointer< vtkPolyData >> &m_meshes);

	void CenterOfMassToOrigin( std::vector< vtkSmartPointer< vtkPolyData >> &m_meshes, bool weighting );

	std::vector< pdm::Sample* >* getGeneratedSamples(std::vector< vtkSmartPointer< vtkPolyData >> &m_meshes); 

	void estimateKernelWidth(std::vector< vtkSmartPointer< vtkPolyData >> datasets, double & EstimatedSigma); 

	void ComputeGeneralizationLoO(std::vector< vtkSmartPointer< vtkPolyData >> trainDatasets, vtkSmartPointer< vtkPolyData > projectPoly, vtkSmartPointer< vtkPolyData > projectGT, RobustKPCA::VectorOfDouble & _generalization);

	double AlignedAndComputeEucliDistPolys(vtkSmartPointer< vtkPolyData > polyTest, vtkSmartPointer< vtkPolyData > polyGT); 

	double ComputeEucliDistPolys(vtkSmartPointer< vtkPolyData > poly1, vtkSmartPointer< vtkPolyData > poly2); 

	void ICPAlignment(vtkSmartPointer< vtkPolyData > _polyGT, vtkSmartPointer< vtkPolyData > & _polyToAlign); 

	void ComputeNormals(vtkPolyData* _poly); 

	vtkSmartPointer< vtkIdList > GetConnectedVertices(vtkSmartPointer< vtkPolyData > _poly, int _id);

	void InsertIdsIntoArray(std::vector< int >  & _ids, vtkSmartPointer< vtkIdList > _connectedVertices);
	

// private parameters 

	RobustKPCA* m_robustKPCA;

	std::string m_recentDir;

	std::vector< vtkSmartPointer< vtkPolyData > > m_corruptedDatasets;  /**> for randomly generated corrupted datasets**/
	std::vector< vtkSmartPointer< vtkPolyData > > m_evalGTDatasets;     /**> for evaluation **/
	std::vector< vtkSmartPointer< vtkPolyData > > m_dataSetsBackup;     /**> for method training use (preserve inputs)**/
	std::vector< vtkSmartPointer< vtkPolyData > > m_testGTForBP;        /**> for back propagation test **/
	std::vector< vtkSmartPointer< vtkPolyData > > m_reconstructions;    /**> for evaluation distance with GT **/

	std::vector< pdm::Sample* >* GTSamples; 

    Ui::NonlinearModellingControls m_Controls;


	//// ******* Add Global Parameters for Model Visualization *********** //// 

	mitk::LookupTableProperty::Pointer m_lookupTableProp;

	std::string m_modelName; 

	RobustKPCA::VectorOfDouble m_modes; 

};

#endif // QmitkAwesomeView_h

