#pragma once 

// ===========================================================================================
//   @ Creation Date:  Feb 2017 
//   @ Created  By  :  Jingting in Fraunhofer IGD 
//
//   @ brief:  this class_RobustKPCA provides several methods imperfect shape reconstruction 
//             (1) Standard PCA 
//             (2) Standard KPCA     
//             (3) Robust PCA solved via ALM             - ICML 2011 Canndes et al. 
//             (4) Robust Kernel Low-rank Representation - NIPS 2016 Xiao et al.
//             (5) Robust Kernel PCA                     - MICCAI 2017 
//
//   @ usage:  - ReadDataMatrix: pass a set of vtkPolys and convert to matrix 
//             - Call functions, get the reconstruction samples 
//                performPCA / RobustPCA 
//                performKPCA 
//                RobustKernelLRR 
//                RobustKernelPCA 
//
//   @ note:   if Robust-KPCA / Robust-KRLRR is called, the m_gramMatrix is replaced by m_robustGramMatrix
//                      as well as all the kernel space statistics
//
//   @ comment : you can find backup codes for NIPS09, PSSV , SVT for RKPCA 
//    
// ============================================================================================= 

#include "pdmModel.h"
#include "pdmModelAbstract.h"
#include "pdmShapeConstrainer.h"
#include "pdmDefaultShapeConstrainer.h"
#include "pdmSample.h"

#include <itkTimeProbe.h>
#include <iostream>

#include <math.h> 
#include <iomanip>

#include <vector> 
#include <algorithm>

#include <J:/MITK_SOURCE/vxl-git-8297f6a04ca79a2e31f8ce31ce3777604dc26258/core/vnl/algo/vnl_svd.h>
#include <J:/MITK_MSVC2013_X64/MITK-2014.10.0_VS12/Eigen-src/Eigen/Core>
#include <J:/MITK_MSVC2013_X64/MITK-2014.10.0_VS12/Eigen-src/Eigen/Eigen>
#include <J:/MITK_MSVC2013_X64/MITK-2014.10.0_VS12/Eigen-src/Eigen/Jacobi>
#include <J:/MITK_MSVC2013_X64/MITK-2014.10.0_VS12/Eigen-src/Eigen/SVD>
#include <J:/MITK_MSVC2013_X64/MITK-2014.10.0_VS12/Eigen-src/Eigen/StdVector>
#include <J:/MITK_MSVC2013_X64/MITK-2014.10.0_VS12/Eigen-src/Eigen/src/Core/Matrix.h>

#include <J:/MITK_MSVC2013_X64/MITK-2014.10.0_VS12/Eigen-src/Eigen/LU>
#include <J:/MITK_MSVC2013_X64/MITK-2014.10.0_VS12/Eigen-src/Eigen/Dense>
#include <J:/MITK_MSVC2013_X64/MITK-2014.10.0_VS12/Eigen-src/Eigen/Eigenvalues>

#include <vnl_symmetric_eigensystem.h> 
#include <vnl_matrix_inverse.h>


// Vtk
#include <vtkPolyData.h>
#include <vtkPolyDataWriter.h>
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
#include <vtkTransformPolyDataFilter.h>
#include <vtkSmartPointer.h>
#include <vtkDataArray.h>
#include <vtkIdList.h>
#include <vtkIdTypeArray.h>

#include <armadillo>
using namespace arma;
using namespace Eigen;
using namespace std;


// inline libs
#include "StatisticalUtilities.h"


class RobustKPCA
{
public:

	typedef vnl_matrix< double > MatrixOfDouble;
	typedef vnl_vector< double > VectorOfDouble;

	RobustKPCA();


	RobustKPCA(double _gamma, bool alignment, bool scaling)
		:m_Utils(nullptr)
	{
		this->m_dataMatrix.fill(0);
		this->m_dataMatrixInput.fill(0); 
		this->m_meanShapeVector.fill(0);
		this->m_inputMeanShapeVector.fill(0);
		this->m_evalGTMatrix.fill(0); 

		constructed_KPCA.fill(0);
		constructed_NIPS09.fill(0); 
		constructed_RobustKernelLRR.fill(0);
		constructed_RobustKernelPCA.fill(0);
		constructed_RobustPCA.fill(0);
		constructed_PCA.fill(0);
		constructed_Miccai17.fill(0); 
		constructed_RKPCA.fill(0); 

		m_numberOfSamples = 0;
		m_numberOfLandmarks = 0;
		m_numberOfRetainedPCs = 0;

		this->m_gramMatrix.fill(0);
		this->m_kernelPCs.fill(0);
		this->m_reducedKernelPCs.fill(0);
		this->m_kPCsMean.fill(0); 
		this->m_kPCsStd.fill(0); 

		this->m_eigenVectors.fill(0);
		this->m_eigenVals.fill(0);
		this->m_eigenVectorsReduced.fill(0); 
		this->m_eigenValsReduced.fill(0); 

		this->m_gamma = _gamma;
		this->m_alignment = alignment; 
		this->m_scaling = scaling;

		this->m_meanShape = NULL;
		this->m_modelType = ""; 
	}


	~RobustKPCA();


	// ================ Call Functions ================================


	/**
	 *  @brief    Align the input training polys without scaling
	 *            Read N vtkPolyDatasets to m_dataMatrix
	 *            Set the mean shape
	 *  @param    m_dataSetsPoly (poly topologies)
	 */
	void ReadDataMatrix(std::vector< vtkSmartPointer< vtkPolyData >> m_polyDataSet); // for each sample( numberOfLandmarks , 3 )


	/**
	 *  @brief   perform PCA Training
	 */
	void performPCA();


	/**
	 *  @brief   perform KPCA internal learn
	 *  @param   get the m_eigenVectors / m_eigenVals
	 *                   m_origGramMatrix / m_origKernelPCs
	 */
	void performKPCA();


	/** 
	 *  @brief   perform Robust Kernel Principal Component Analysis propsoed in NIPS2009 
	 *  @param   the only difference with KPCA is the reconstruction function 
	 *           use backProjectRobust
	 */
	void performNIPS09(); 


	/**
	 *  @brief    Robust PCA + ALM
	 *            minimizing the nuclear norm of low-rank matrix + l1 norm of sparse matrix
	 *            solved by inexact augmented Lagrange Multiplier proposed in NIPS 2011
	 *  @input:   m_dataMatrix
	 *  @return:  constructed_RobustPCA
	 */
	void RobustPCA();


	/**
	 * @brief   RKPCA 
	 */
	void RKPCA(int _iterNum, double _proportion);


	/**
	 *  @brief   Miccai_RKPCA 
	 */
	void Miccai17(); 


	/**
	 *  @brief     perform Robust Kernelized PCA (Xiao et al. 2016)
	 *  (1) Kernelize LRR - Perform Kernel PCA on input data matrix: Get K and SVD(K)
	 *  (2) Robust KLRR - Phi = Phi*L + S: Get L which represents data composition
	 *  (3) Construct data: return XL
	 *
	 *  @input:     dataMatrix
	 *              _gamma for kernel PCA
	 *  @return:    Standard Kernel Matrix
	 *              Robust Kernel Matrix
	 */
	void RobustKernelLRR();


	/**
	 *  @brief   Get shape reconstruction with defined matrix and parameter
	 *  @param   input the kernel PC with the dimension of reduced KPCs
	 *           output the reconstructed shape vector = m_numberOflandmarks x 3
	 */
	VectorOfDouble GetBackProjection(VectorOfDouble parameter);


	/**
	 *  @brief  Get the projected shape from the nonlinear model which is aligned to the original location in physical shape space
	 *          Align the input shape with m_dataSetPoly before projection
	 *  @param  input the shape to be project
	 *          output the aligned projected shape by procrustesAlignment
	 */
	vtkSmartPointer< vtkPolyData > ProjectShapeNonlinearModel(double _proportion, vtkSmartPointer< vtkPolyData > inputShape, int _modes);


	/**
	 *  @brief  Get the projected shape from the linear model without extra alignment
	 *          Since this is linear model, extra alignment is not required
	 *  @param  input the shape to be projected
	 *          output the projected shape to m_Utils;
	 */
	vtkSmartPointer< vtkPolyData > ProjectShapeLinearModel(double _proportion, vtkSmartPointer< vtkPolyData > inputShape, int _modes);


	/**
	 *  @brief    get randomly sampled data matrix 
	 */
	std::vector< vtkSmartPointer< vtkPolyData > > GetRandomCorruptedDataMatrix(); 


	// ================ Model - Evaluation ==============================

	/**
	 *  @brief   Return the Generalization ability & Speciticity of Linear model 
	 *  @param   input preserved number of modes 
	 *           input number of samples (in total)
	 */
	void ComputeLinearModelGSWithPreservedModes(int mode, int numSampling, double& _specifcity, VectorOfDouble & _specVec); 


	/**
	 *  @brief   Return the Generalization ability & Speciticity of Linear model
	 *  @param   input preserved number of modes
	 *           input number of samples (in total)
	 */
	void ComputeNonlinearModelGSWithPreservedModes(int mode, int numSampling, double& _specifcity, VectorOfDouble & _specVec);


	/**
	 *  @brief   for evaluation use, align the ground truth with training datasets 
	 */
	void AlignEvalGTWithTrainingMatrix(std::vector< vtkSmartPointer< vtkPolyData >> & m_evalGTPolys); 
	

	// ================ Get Functions ===================================

	/**
	 * @brief  Get the aligned mean shape vector from different methods
	 */
	VectorOfDouble GetMeanShapeVector() const
	{
		return m_meanShapeVector;
	}

	vtkSmartPointer< vtkPolyData > GetMeanShape() const
	{
		this->m_meanShape->DeepCopy(this->m_dataSetPoly.at(0));

		vtkPoints* points = this->m_meanShape->GetPoints();
		for (int i = 0; i < points->GetNumberOfPoints() * 3; i += 3)
		{
			points->SetPoint(i / 3, this->m_meanShapeVector[i], this->m_meanShapeVector[i + 1], this->m_meanShapeVector[i + 2]);
		}
		points->Modified();
		this->m_meanShape->Modified();

		return this->m_meanShape;
	}

	MatrixOfDouble GetGramMatrix() const
	{
		//std::cout << " If RKPCA is called, the m_gramMatrix is the robust Gram Matrix, as well as variances" << std::endl; 
		return this->m_gramMatrix;
	}

	MatrixOfDouble GetEigenVectors() const
	{
		//std::cout << " return the reduced eigen vectors with dimension: " << this->m_eigenVectorsReduced.rows() << " x " << this->m_eigenVectorsReduced.cols() << std::endl; 
		return this->m_eigenVectorsReduced;
	}

	VectorOfDouble GetEigenValues() const
	{
		//std::cout << " return the reduced eigen values " << this->m_eigenValsReduced.size() << std::endl;
		return this->m_eigenValsReduced;
	}

	unsigned int GetParameterCount() const
	{
		//std::cout << " return " << this->m_numberOfRetainedPCs << " kernel PCs covered 97% variances by performing KPCA on data input matrix ...  " << std::endl;
		return this->m_numberOfRetainedPCs;
	}

	MatrixOfDouble GetReducedKernelPCs() const
	{
		return this->m_reducedKernelPCs;
	}

	StatisticalUtilities* GetPCAUtils() const
	{
		return this->m_Utils;
	}

	VectorOfDouble GetKPCsMean() const
	{
		return this->m_kPCsMean; 
	}

	VectorOfDouble GetKPCsStd() const
	{
		return this->m_kPCsStd; 
	}

	std::string GetModelType() const
	{
		return this->m_modelType; 
	}


	// ================ Inline Functions ================================


	std::vector< vtkSmartPointer< vtkPolyData>> getConstructed_KPCA();

	std::vector< vtkSmartPointer< vtkPolyData>> getConstructed_NIPS09(); 

	std::vector< vtkSmartPointer< vtkPolyData>> getConstructed_RobustKernelLRR();

//	std::vector< vtkSmartPointer< vtkPolyData>> getConstructed_RobustKernelPCA();

	std::vector< vtkSmartPointer< vtkPolyData>> getConstructed_RobustPCA();

	std::vector< vtkSmartPointer< vtkPolyData>> getConstructed_PCA();

	std::vector< vtkSmartPointer< vtkPolyData>> getConstructed_Miccai17(); 

	std::vector< vtkSmartPointer< vtkPolyData>> getConstructed_RKPCA(); 

	std::vector< vtkSmartPointer< vtkPolyData>> getTrainingPolys(); 


	/**
	 *  @brief    compute Eigen-decomposition of a matrix 
	 *            X = Q A Q^-1 
	 */
	void eigenDecomposition(MatrixOfDouble _dataMatrix, MatrixOfDouble & _eigenVectors, VectorOfDouble & _eigenValues); 


	/**
	 *  @brief   compute Singular Value Decomposition of a matrix 
	 *           X = U W V^T 
	 */
	void singularValueDecomposition(MatrixOfDouble _dataMatrix, MatrixOfDouble & _leftColumns, MatrixOfDouble & _svds, MatrixOfDouble & _rightColumns); 


private:


	/**
	 *  @brief   get the shape reconstruction with the input parameter (kernel principal components)
  	 *  @param   _parameter : the kernel principal components with the size = m_numberOfRetainedPCs
	 *           implementation according Wang et al. 2012
	 */
	VectorOfDouble backProject(VectorOfDouble inputKPCs);


	/**
	 *  @brief   get the shape reconstruction with the input parameter and input shape vector 
	 *  @param   the method comes from NIPS2009 
	 *           shape vector and shape parameter KPCs are input 
	 *  @param   input       : Kernel Principal Components 
	 *           shapeVector : 
	 */
	VectorOfDouble backProjectRobust(VectorOfDouble inputKPCs, VectorOfDouble shapeVector); 


	/**
	 *  @brief   my own back projection 
	 *           Minimize: original projection and distance with all input datasets 
	 */
	VectorOfDouble backProjectAwesome(VectorOfDouble _inputKPCs, VectorOfDouble _inputShape, double _constBP);


	/**
	 *  @brief  back projection of NIPS 2009 
	 */
	VectorOfDouble backProjectNIPS09(VectorOfDouble _shapeSample, double _constBP); 


	/**
	 *  @brief   compute the projected shape vector of PCA models via dimensionality reduction 
	 *           b = N^T(X - \bar(X)) 
	 *           X = \bar(X) + Nb 
	 *  @param   m_modes = the number of retained modes in the projection for evaluation 
	 */
	VectorOfDouble backProjectLinear(VectorOfDouble shapeVector, int m_mode); 



	VectorOfDouble backProjectInternal(VectorOfDouble inputKPCs, MatrixOfDouble dataMatrix, MatrixOfDouble eigenVecs, VectorOfDouble initVector);


	/**
	 *  @brief   compute \hat(X) based on shape vector 
	 *  @param   input  : current data matrix, low-rank kernel matrix, svd.U() -> eigenvectors 
	 *           output : reconstructed \hat(X) 
	 */
	MatrixOfDouble computeGradientLowRankX(MatrixOfDouble _inputData, MatrixOfDouble _kernelMatrix); 


	/**
	 *  @brief   compute partial gradient of K / X 
	 *  @param   input  : current data matrix, low-rank kernel matrix 
	 *           output : subgradient \partial(K) / \partial(X) 
	 */
	MatrixOfDouble subgradientKernelMatrix(MatrixOfDouble _inputData, MatrixOfDouble _kernelMatrix); 


	/**
	 *  @brief   low-rank kernel matrix via inexact ALM 
	 *  @param   return: the reconstructed Low-rank Matrix 
	 *           input: current kernel matrix, multiplier, current low-rank kernel matrix 
	 */
	MatrixOfDouble LowRankModelingKernelMatrix(MatrixOfDouble _inputData, MatrixOfDouble _kernelMatrix, MatrixOfDouble & m_K); 


	// ================ Utilities =======================================


	/**
	 *  @brief   get squared Euclidean distance of two poly vectors 
	 *           dist = sqrt( pow(px1-px2, 2) + pow(py1-py2, 2) + pow(pz1-pz2, 2) )
	 */
	double getEuclideanDistanceOfVectors(VectorOfDouble vec1, VectorOfDouble vec2); 


	void CenterOfMassToOrigin(std::vector< vtkSmartPointer< vtkPolyData >> &m_meshes);


	void AlignDataSetsWithoutScaling(std::vector< vtkSmartPointer< vtkPolyData >>& meshes);


	void AlignDataSetsWithScaling(std::vector< vtkSmartPointer< vtkPolyData >> &m_meshes); 


	void RemoveEntries(double _proportion, vtkSmartPointer< vtkPolyData > & _mesh); 


	vtkSmartPointer< vtkIdList > GetConnectedVertices(vtkSmartPointer< vtkPolyData > poly, int id); 


	void InsertIdsIntoArray(std::vector< int > & _ids, vtkSmartPointer< vtkIdList > _connectedVertices); 


	unsigned int m_numberOfSamples;
	unsigned int m_numberOfLandmarks;
	unsigned int m_numberOfRetainedPCs;

	double m_gamma;


	MatrixOfDouble m_dataMatrix;           /**>  Data Matrix: stack of column vectors  **/
	MatrixOfDouble m_dataMatrixInput;      /**>  Data Matrix: read from input polys    **/
	VectorOfDouble m_meanShapeVector;      /**>  mean shape calculation in ReadDataMatrix **/

	MatrixOfDouble m_gramMatrix;           /**> kernel matrix **/
	MatrixOfDouble m_kernelPCs;            /**> Kernel Principal Components from m_gramMatrix - stack of rows **/
	MatrixOfDouble m_reducedKernelPCs;     /**> Reduced Kernel Principal Components from m_origGramMatrix - stack of rows **/
	VectorOfDouble m_kPCsMean;             /**> mean reduced kernel principal components  **/
	VectorOfDouble m_kPCsStd;              /**> std reduced kernel principal components  **/

	MatrixOfDouble m_eigenVectors;         /**> normalized **/
	VectorOfDouble m_eigenVals;

	MatrixOfDouble m_eigenVectorsReduced;  /**> reduced eigenVectors (normalized), stack of row vectors **/
	VectorOfDouble m_eigenValsReduced;


	////////////////////////////////////////////////////////////////////////////////////////

	MatrixOfDouble constructed_PCA;               /**> constructed samples by PCA **/
	MatrixOfDouble constructed_RobustPCA;         /**> Constructed Data Matrix of RobustPCA **/
	MatrixOfDouble constructed_KPCA;              /**> Constructed pre-images using KPCA **/
	MatrixOfDouble constructed_NIPS09;            /**> Constructed RKPCA from NIPS09 **/
	MatrixOfDouble constructed_RobustKernelLRR;   /**> Constructed Data Matrix of RobustKernelLRR **/
	MatrixOfDouble constructed_RobustKernelPCA;   /**> Constructed low-rank representation **/
	MatrixOfDouble constructed_Miccai17;          /**> Constructed Miccai-17 **/
	MatrixOfDouble constructed_RKPCA;               /**> Constructed Backup **/

	VectorOfDouble m_inputMeanShapeVector;        /**> the mean shape vector from the input datasets **/
	MatrixOfDouble m_evalGTMatrix;                /**> the matrix for aligned ground truth datasets **/

	std::vector< vtkSmartPointer< vtkPolyData >> m_dataSetPoly;   /**> aligned training datasets **/

	vtkSmartPointer< vtkPolyData > m_meanShape;   /**> the mean shape **/

	StatisticalUtilities* m_Utils;                /**> linear model statistics **/

	bool m_alignment;                             /**> default true: note that false is for pdf sampling **/
	bool m_scaling;                               /**> true: align the datasets before modelling **/

	std::vector< vtkSmartPointer< vtkLandmarkTransform> > m_landmarkTransformVector; /**> Preserves transformation for reconstruction **/

	std::string m_modelType; 

};

// ======================================================================


inline void RobustKPCA::AlignDataSetsWithoutScaling(std::vector< vtkSmartPointer< vtkPolyData >> &m_meshes)
{
	/* Procrustes Alignment without Scaling */
	vtkSmartPointer< vtkMultiBlockDataSet > inputMultiBlock = vtkSmartPointer< vtkMultiBlockDataSet >::New();
	vtkSmartPointer< vtkMultiBlockDataSet > outputMultiBlock = vtkSmartPointer< vtkMultiBlockDataSet >::New();
	vtkSmartPointer< vtkProcrustesAlignmentFilter > procrustes = vtkSmartPointer< vtkProcrustesAlignmentFilter >::New();

	int sampleCnt = 0;

	for (unsigned int i = 0; i < m_meshes.size(); i++)
	{
		inputMultiBlock->SetBlock(sampleCnt, m_meshes.at(i));
		sampleCnt++;
	}

	procrustes->SetInputData(inputMultiBlock);
	procrustes->SetStartFromCentroid(false);
	procrustes->GetLandmarkTransform()->SetModeToRigidBody();

	procrustes->Update();

	outputMultiBlock = procrustes->GetOutput();

	for (unsigned int i = 0; i < m_meshes.size(); i++)
	{
		vtkPolyData* outputPoly = dynamic_cast< vtkPolyData* >(outputMultiBlock->GetBlock(i));

		m_meshes.at(i)->DeepCopy(outputPoly);
	}
}


inline void RobustKPCA::AlignDataSetsWithScaling(std::vector< vtkSmartPointer< vtkPolyData >> &m_meshes)
{
	cout << __FUNCTION__ << " started : ";
	/* AVERAGE_UNIT_DISTANCE_SCALING : shapes are scaled that the average distance to the origin is 1 */

	double meanScaling = 0.0;

	for (unsigned int i = 0; i < m_meshes.size(); i++)
	{
		double dScalingFac = 0.0;
		long nLandmarks = m_meshes.at(i)->GetNumberOfPoints();

		for (unsigned int j = 0; j < nLandmarks; j++)
		{
			double dNorm = 0.0;
			double p[3];

			m_meshes.at(i)->GetPoint(j, p);

			for (unsigned int k = 0; k < 3; k++)
				dNorm += p[k] * p[k];

			dScalingFac += sqrt(dNorm);
		}

		dScalingFac /= nLandmarks;
		meanScaling += dScalingFac;

		for (unsigned int j = 0; j < nLandmarks; j++)
		{
			double p[3];
			m_meshes.at(i)->GetPoint(j, p);

			for (unsigned int k = 0; k < 3; k++)
				p[k] /= dScalingFac;

			m_meshes.at(i)->GetPoints()->SetPoint(j, p);
		}
	}

	meanScaling /= m_meshes.size();
	cout << " meanScaling = " << meanScaling;

	/* Procrustes Alignment without Scaling */
	vtkSmartPointer< vtkMultiBlockDataSet > inputMultiBlock = vtkSmartPointer< vtkMultiBlockDataSet >::New();
	vtkSmartPointer< vtkMultiBlockDataSet > outputMultiBlock = vtkSmartPointer< vtkMultiBlockDataSet >::New();
	vtkSmartPointer< vtkProcrustesAlignmentFilter > procrustes = vtkSmartPointer< vtkProcrustesAlignmentFilter >::New();

	int sampleCnt = 0;

	for (unsigned int i = 0; i < m_meshes.size(); i++)
	{
		inputMultiBlock->SetBlock(sampleCnt, m_meshes.at(i));
		sampleCnt++;
	}

	procrustes->SetInputData(inputMultiBlock);
	procrustes->GetLandmarkTransform()->SetModeToRigidBody();
	procrustes->Update();

	outputMultiBlock = procrustes->GetOutput();

	for (unsigned int i = 0; i < m_meshes.size(); i++)
	{
		vtkPolyData* outputPoly = dynamic_cast< vtkPolyData* >(outputMultiBlock->GetBlock(i));

		m_meshes.at(i)->DeepCopy(outputPoly);
	}

	cout << " ... done " << endl;
}


inline void RobustKPCA::CenterOfMassToOrigin(std::vector< vtkSmartPointer< vtkPolyData >> &m_meshes)
{
	cout << __FUNCTION__ << " Start ... ";
	for (unsigned int ns = 0; ns < m_meshes.size(); ns++)
	{
		vtkSmartPointer< vtkCenterOfMass > centerOfMassFilter = vtkSmartPointer< vtkCenterOfMass >::New();

		double center[3];

		centerOfMassFilter->SetInputData(m_meshes.at(ns));
		centerOfMassFilter->SetUseScalarsAsWeights(false);
		centerOfMassFilter->Update();
		centerOfMassFilter->GetCenter(center);
		
		for (unsigned int np = 0; np < m_meshes.at(ns)->GetNumberOfPoints(); np++)
		{
			double point_old[3];
			m_meshes.at(ns)->GetPoint(np, point_old);

			double point_new[3];
			point_new[0] = point_old[0] - center[0];
			point_new[1] = point_old[1] - center[1];
			point_new[2] = point_old[2] - center[2];
			m_meshes.at(ns)->GetPoints()->SetPoint(np, point_new);
		}
	}

	cout << " done . " << endl;
}


inline std::vector< vtkSmartPointer< vtkPolyData >> RobustKPCA::getConstructed_KPCA()
{
	std::vector< pdm::Sample* >* Samples = new std::vector< pdm::Sample* >;

	for (int i = 0; i < constructed_KPCA.cols(); i++)
	{
		pdm::Sample* sample = new pdm::Sample(m_numberOfLandmarks, 3);

		for (int j = 0; j < constructed_KPCA.rows(); j += 3)
		{
			double pos[3];

			pos[0] = constructed_KPCA[j][i];
			pos[1] = constructed_KPCA[j + 1][i];
			pos[2] = constructed_KPCA[j + 2][i];

			sample->SetLandmark(j / 3, pos);
		}

		Samples->push_back(sample);
	}

	std::vector< vtkSmartPointer< vtkPolyData >> backPolys;
	for (int i = 0; i < this->m_numberOfSamples; i++)
	{
		vtkSmartPointer< vtkPolyData > poly = vtkSmartPointer< vtkPolyData >::New();
		poly->DeepCopy(this->m_dataSetPoly.at(i));
		backPolys.push_back(poly);
	}

	for (int i = 0; i < this->m_numberOfSamples; i++)
	{
		pdm::Sample* sample = new pdm::Sample(Samples->at(i)->GetNumberOfLandmarks(), 3);
		sample = Samples->at(i);

		for (int j = 0; j < sample->GetNumberOfLandmarks(); j++)
		{
			pdm::VectorOfDouble pos(3); pos.fill(0.0);
			sample->GetLandmark(j, pos);

			double coords[3];
			coords[0] = pos[0]; coords[1] = pos[1]; coords[2] = pos[2];

			backPolys.at(i)->GetPoints()->SetPoint(j, coords);
		}
	}

	/*for (int i = 0; i < backPolys.size(); i++)
	{
		vtkSmartPointer< vtkTransformPolyDataFilter > transformFilter = vtkSmartPointer< vtkTransformPolyDataFilter >::New();
		transformFilter->SetInputData(backPolys.at(i));
		transformFilter->SetTransform(m_landmarkTransformVector.at(i));
		transformFilter->Update();
		backPolys.at(i)->SetPoints(transformFilter->GetOutput()->GetPoints());
	}*/

	return backPolys;
}


inline std::vector< vtkSmartPointer< vtkPolyData >> RobustKPCA::getConstructed_NIPS09()
{
	std::vector< pdm::Sample* >* Samples = new std::vector< pdm::Sample* >;

	for (int i = 0; i < constructed_NIPS09.cols(); i++)
	{
		pdm::Sample* sample = new pdm::Sample(m_numberOfLandmarks, 3);

		for (int j = 0; j < constructed_NIPS09.rows(); j += 3)
		{
			double pos[3];

			pos[0] = constructed_NIPS09[j][i];
			pos[1] = constructed_NIPS09[j + 1][i];
			pos[2] = constructed_NIPS09[j + 2][i];

			sample->SetLandmark(j / 3, pos);
		}

		Samples->push_back(sample);
	}

	std::vector< vtkSmartPointer< vtkPolyData >> backPolys;
	for (int i = 0; i < this->m_numberOfSamples; i++)
	{
		vtkSmartPointer< vtkPolyData > poly = vtkSmartPointer< vtkPolyData >::New();
		poly->DeepCopy(this->m_dataSetPoly.at(i));
		backPolys.push_back(poly);
	}

	for (int i = 0; i < this->m_numberOfSamples; i++)
	{
		pdm::Sample* sample = new pdm::Sample(Samples->at(i)->GetNumberOfLandmarks(), 3);
		sample = Samples->at(i);

		for (int j = 0; j < sample->GetNumberOfLandmarks(); j++)
		{
			pdm::VectorOfDouble pos(3); pos.fill(0.0);
			sample->GetLandmark(j, pos);

			double coords[3];
			coords[0] = pos[0]; coords[1] = pos[1]; coords[2] = pos[2];

			backPolys.at(i)->GetPoints()->SetPoint(j, coords);
		}
	}

	/*for (int i = 0; i < backPolys.size(); i++)
	{
		vtkSmartPointer< vtkTransformPolyDataFilter > transformFilter = vtkSmartPointer< vtkTransformPolyDataFilter >::New();
		transformFilter->SetInputData(backPolys.at(i));
		transformFilter->SetTransform(m_landmarkTransformVector.at(i));
		transformFilter->Update();
		backPolys.at(i)->SetPoints(transformFilter->GetOutput()->GetPoints());
	}*/

	return backPolys;
}


inline std::vector< vtkSmartPointer< vtkPolyData >> RobustKPCA::getConstructed_RobustKernelLRR()
{
	std::vector< pdm::Sample* >* Samples = new std::vector< pdm::Sample* >;

	for (int i = 0; i < constructed_RobustKernelLRR.cols(); i++)
	{
		pdm::Sample* sample = new pdm::Sample(m_numberOfLandmarks, 3);

		for (int j = 0; j < constructed_RobustKernelLRR.rows(); j += 3)
		{
			double pos[3];

			pos[0] = constructed_RobustKernelLRR[j][i];
			pos[1] = constructed_RobustKernelLRR[j + 1][i];
			pos[2] = constructed_RobustKernelLRR[j + 2][i];

			sample->SetLandmark(j / 3, pos);
		}

		Samples->push_back(sample);
	}

	std::vector< vtkSmartPointer< vtkPolyData >> backPolys;
	for (int i = 0; i < this->m_numberOfSamples; i++)
	{
		vtkSmartPointer< vtkPolyData > poly = vtkSmartPointer< vtkPolyData >::New();
		poly->DeepCopy(this->m_dataSetPoly.at(i));
		backPolys.push_back(poly);
	}

	for (int i = 0; i < this->m_numberOfSamples; i++)
	{
		pdm::Sample* sample = new pdm::Sample(Samples->at(i)->GetNumberOfLandmarks(), 3);
		sample = Samples->at(i);

		for (int j = 0; j < sample->GetNumberOfLandmarks(); j++)
		{
			pdm::VectorOfDouble pos(3); pos.fill(0.0);
			sample->GetLandmark(j, pos);

			double coords[3];
			coords[0] = pos[0]; coords[1] = pos[1]; coords[2] = pos[2];

			backPolys.at(i)->GetPoints()->SetPoint(j, coords);
		}
	}

	/*for (int i = 0; i < backPolys.size(); i++)
	{
		vtkSmartPointer< vtkTransformPolyDataFilter > transformFilter = vtkSmartPointer< vtkTransformPolyDataFilter >::New();
		transformFilter->SetInputData(backPolys.at(i));
		transformFilter->SetTransform(m_landmarkTransformVector.at(i));
		transformFilter->Update();
		backPolys.at(i)->SetPoints(transformFilter->GetOutput()->GetPoints());
	}*/

	return backPolys;
}


//inline std::vector< vtkSmartPointer< vtkPolyData >> RobustKPCA::getConstructed_RobustKernelPCA()
//{
//	////////// disorder back the m_dataMatrix //////////////////////////////
//
//	/*for (int col = 0; col < this->constructed_RobustKernelPCA.cols(); col++)
//	{
//		for (int row = 0; row < this->constructed_RobustKernelPCA.rows(); row += 2)
//		{
//			std::swap(this->constructed_RobustKernelPCA[row][col], this->constructed_RobustKernelPCA[row + 1][col]);
//		}
//	}*/
//
//	///////////////////////////////////////////////////////////////////////
//
//
//	std::vector< pdm::Sample* >* Samples = new std::vector< pdm::Sample* >;
//
//	for (int i = 0; i < this->constructed_RobustKernelPCA.cols(); i++)
//	{
//		pdm::Sample* sample = new pdm::Sample(m_numberOfLandmarks, 3);
//
//		for (int j = 0; j < this->constructed_RobustKernelPCA.rows(); j += 3)
//		{
//			double pos[3];
//
//			pos[0] = this->constructed_RobustKernelPCA[j][i];
//			pos[1] = this->constructed_RobustKernelPCA[j + 1][i];
//			pos[2] = this->constructed_RobustKernelPCA[j + 2][i];
//
//			sample->SetLandmark(j / 3, pos);
//		}
//
//		Samples->push_back(sample);
//	}
//
//	std::vector< vtkSmartPointer< vtkPolyData >> backPolys;
//	for (int i = 0; i < this->m_numberOfSamples; i++)
//	{
//		vtkSmartPointer< vtkPolyData > poly = vtkSmartPointer< vtkPolyData >::New();
//		poly->DeepCopy(this->m_dataSetPoly.at(i));
//		backPolys.push_back(poly);
//	}
//
//	for (int i = 0; i < this->m_numberOfSamples; i++)
//	{
//		pdm::Sample* sample = new pdm::Sample(Samples->at(i)->GetNumberOfLandmarks(), 3);
//		sample = Samples->at(i);
//
//		for (int j = 0; j < sample->GetNumberOfLandmarks(); j++)
//		{
//			pdm::VectorOfDouble pos(3); pos.fill(0.0);
//			sample->GetLandmark(j, pos);
//
//			double coords[3];
//			coords[0] = pos[0]; coords[1] = pos[1]; coords[2] = pos[2];
//
//			backPolys.at(i)->GetPoints()->SetPoint(j, coords);
//		}
//	}
//
//	/*for (int i = 0; i < backPolys.size(); i++)
//	{
//		vtkSmartPointer< vtkTransformPolyDataFilter > transformFilter = vtkSmartPointer< vtkTransformPolyDataFilter >::New();
//		transformFilter->SetInputData(backPolys.at(i));
//		transformFilter->SetTransform(m_landmarkTransformVector.at(i));
//		transformFilter->Update();
//		backPolys.at(i)->SetPoints(transformFilter->GetOutput()->GetPoints());
//	}*/
//
//	return backPolys;
//}


inline std::vector< vtkSmartPointer< vtkPolyData >> RobustKPCA::getConstructed_RobustPCA()
{
	std::vector< pdm::Sample* >* Samples = new std::vector< pdm::Sample* >;

	for (int i = 0; i < constructed_RobustPCA.cols(); i++)
	{
		pdm::Sample* sample = new pdm::Sample(m_numberOfLandmarks, 3);

		for (int j = 0; j < constructed_RobustPCA.rows(); j += 3)
		{
			double pos[3];

			pos[0] = constructed_RobustPCA[j][i];
			pos[1] = constructed_RobustPCA[j + 1][i];
			pos[2] = constructed_RobustPCA[j + 2][i];

			sample->SetLandmark(j / 3, pos);
		}

		Samples->push_back(sample);
	}

	std::vector< vtkSmartPointer< vtkPolyData >> backPolys;
	for (int i = 0; i < this->m_numberOfSamples; i++)
	{
		vtkSmartPointer< vtkPolyData > poly = vtkSmartPointer< vtkPolyData >::New();
		poly->DeepCopy(this->m_dataSetPoly.at(i));
		backPolys.push_back(poly);
	}

	for (int i = 0; i < this->m_numberOfSamples; i++)
	{
		pdm::Sample* sample = new pdm::Sample(Samples->at(i)->GetNumberOfLandmarks(), 3);
		sample = Samples->at(i);

		for (int j = 0; j < sample->GetNumberOfLandmarks(); j++)
		{
			pdm::VectorOfDouble pos(3); pos.fill(0.0);
			sample->GetLandmark(j, pos);

			double coords[3];
			coords[0] = pos[0]; coords[1] = pos[1]; coords[2] = pos[2];

			backPolys.at(i)->GetPoints()->SetPoint(j, coords);
		}
	}

	/*for (int i = 0; i < backPolys.size(); i++)
	{
		vtkSmartPointer< vtkTransformPolyDataFilter > transformFilter = vtkSmartPointer< vtkTransformPolyDataFilter >::New();
		transformFilter->SetInputData(backPolys.at(i));
		transformFilter->SetTransform(m_landmarkTransformVector.at(i));
		transformFilter->Update();
		backPolys.at(i)->SetPoints(transformFilter->GetOutput()->GetPoints());
	}*/

	return backPolys;
}


inline std::vector< vtkSmartPointer< vtkPolyData >> RobustKPCA::getConstructed_PCA()
{
	std::vector< pdm::Sample* >* Samples = new std::vector< pdm::Sample* >;

	for (int i = 0; i < this->constructed_PCA.cols(); i++)
	{
		pdm::Sample* sample = new pdm::Sample(m_numberOfLandmarks, 3);

		for (int j = 0; j < this->constructed_PCA.rows(); j += 3)
		{
			double pos[3];

			pos[0] = this->constructed_PCA[j][i];
			pos[1] = this->constructed_PCA[j + 1][i];
			pos[2] = this->constructed_PCA[j + 2][i];

			sample->SetLandmark(j / 3, pos);
		}

		Samples->push_back(sample);
	}

	std::vector< vtkSmartPointer< vtkPolyData >> backPolys;
	for (int i = 0; i < this->m_numberOfSamples; i++)
	{
		vtkSmartPointer< vtkPolyData > poly = vtkSmartPointer< vtkPolyData >::New();
		poly->DeepCopy(this->m_dataSetPoly.at(i));
		backPolys.push_back(poly);
	}

	for (int i = 0; i < this->m_numberOfSamples; i++)
	{
		pdm::Sample* sample = new pdm::Sample(Samples->at(i)->GetNumberOfLandmarks(), 3);
		sample = Samples->at(i);

		for (int j = 0; j < sample->GetNumberOfLandmarks(); j++)
		{
			pdm::VectorOfDouble pos(3); pos.fill(0.0);
			sample->GetLandmark(j, pos);

			double coords[3];
			coords[0] = pos[0]; coords[1] = pos[1]; coords[2] = pos[2];

			backPolys.at(i)->GetPoints()->SetPoint(j, coords);
		}
	}

	for (int i = 0; i < backPolys.size(); i++)
	{
		vtkSmartPointer< vtkTransformPolyDataFilter > transformFilter = vtkSmartPointer< vtkTransformPolyDataFilter >::New();
		transformFilter->SetInputData(backPolys.at(i));
		transformFilter->SetTransform(m_landmarkTransformVector.at(i));
		transformFilter->Update();
		backPolys.at(i)->SetPoints(transformFilter->GetOutput()->GetPoints());
	}

	return backPolys;
}


inline std::vector< vtkSmartPointer< vtkPolyData >> RobustKPCA::getConstructed_Miccai17()
{
	std::vector< pdm::Sample* >* Samples = new std::vector< pdm::Sample* >;

	for (int i = 0; i < this->constructed_Miccai17.cols(); i++)
	{
		pdm::Sample* sample = new pdm::Sample(m_numberOfLandmarks, 3);

		for (int j = 0; j < this->constructed_Miccai17.rows(); j += 3)
		{
			double pos[3];

			pos[0] = this->constructed_Miccai17[j][i];
			pos[1] = this->constructed_Miccai17[j + 1][i];
			pos[2] = this->constructed_Miccai17[j + 2][i];

			sample->SetLandmark(j / 3, pos);
		}

		Samples->push_back(sample);
	}

	std::vector< vtkSmartPointer< vtkPolyData >> backPolys;
	for (int i = 0; i < this->m_numberOfSamples; i++)
	{
		vtkSmartPointer< vtkPolyData > poly = vtkSmartPointer< vtkPolyData >::New();
		poly->DeepCopy(this->m_dataSetPoly.at(i));
		backPolys.push_back(poly);
	}

	for (int i = 0; i < this->m_numberOfSamples; i++)
	{
		pdm::Sample* sample = new pdm::Sample(Samples->at(i)->GetNumberOfLandmarks(), 3);
		sample = Samples->at(i);

		for (int j = 0; j < sample->GetNumberOfLandmarks(); j++)
		{
			pdm::VectorOfDouble pos(3); pos.fill(0.0);
			sample->GetLandmark(j, pos);

			double coords[3];
			coords[0] = pos[0]; coords[1] = pos[1]; coords[2] = pos[2];

			backPolys.at(i)->GetPoints()->SetPoint(j, coords);
		}
	}

	for (int i = 0; i < backPolys.size(); i++)
	{
		vtkSmartPointer< vtkTransformPolyDataFilter > transformFilter = vtkSmartPointer< vtkTransformPolyDataFilter >::New();
		transformFilter->SetInputData(backPolys.at(i));
		transformFilter->SetTransform(m_landmarkTransformVector.at(i));
		transformFilter->Update();
		backPolys.at(i)->SetPoints(transformFilter->GetOutput()->GetPoints());
	}

	return backPolys;
}


inline std::vector< vtkSmartPointer< vtkPolyData >> RobustKPCA::getConstructed_RKPCA()
{
	std::vector< pdm::Sample* >* Samples = new std::vector< pdm::Sample* >;

	for (int i = 0; i < this->constructed_RKPCA.cols(); i++)
	{
		pdm::Sample* sample = new pdm::Sample(m_numberOfLandmarks, 3);

		for (int j = 0; j < this->constructed_RKPCA.rows(); j += 3)
		{
			double pos[3];

			pos[0] = this->constructed_RKPCA[j][i];
			pos[1] = this->constructed_RKPCA[j + 1][i];
			pos[2] = this->constructed_RKPCA[j + 2][i];

			sample->SetLandmark(j / 3, pos);
		}

		Samples->push_back(sample);
	}

	std::vector< vtkSmartPointer< vtkPolyData >> backPolys;
	for (int i = 0; i < this->m_numberOfSamples; i++)
	{
		vtkSmartPointer< vtkPolyData > poly = vtkSmartPointer< vtkPolyData >::New();
		poly->DeepCopy(this->m_dataSetPoly.at(i));
		backPolys.push_back(poly);
	}

	for (int i = 0; i < this->m_numberOfSamples; i++)
	{
		pdm::Sample* sample = new pdm::Sample(Samples->at(i)->GetNumberOfLandmarks(), 3);
		sample = Samples->at(i);

		for (int j = 0; j < sample->GetNumberOfLandmarks(); j++)
		{
			pdm::VectorOfDouble pos(3); pos.fill(0.0);
			sample->GetLandmark(j, pos);

			double coords[3];
			coords[0] = pos[0]; coords[1] = pos[1]; coords[2] = pos[2];

			backPolys.at(i)->GetPoints()->SetPoint(j, coords);
		}
	}

	for (int i = 0; i < backPolys.size(); i++)
	{
		vtkSmartPointer< vtkTransformPolyDataFilter > transformFilter = vtkSmartPointer< vtkTransformPolyDataFilter >::New();
		transformFilter->SetInputData(backPolys.at(i));
		transformFilter->SetTransform(m_landmarkTransformVector.at(i));
		transformFilter->Update();
		backPolys.at(i)->SetPoints(transformFilter->GetOutput()->GetPoints());
	}

	return backPolys;
}


inline std::vector< vtkSmartPointer< vtkPolyData >> RobustKPCA::getTrainingPolys()
{
	return this->m_dataSetPoly; 
}


inline std::vector< vtkSmartPointer< vtkPolyData >> RobustKPCA::GetRandomCorruptedDataMatrix()
{
	std::vector< pdm::Sample* >* Samples = new std::vector< pdm::Sample* >;

	for (int i = 0; i < this->m_dataMatrix.cols(); i++)
	{
		pdm::Sample* sample = new pdm::Sample(m_numberOfLandmarks, 3);

		for (int j = 0; j < this->m_dataMatrix.rows(); j += 3)
		{
			double pos[3];

			pos[0] = this->m_dataMatrix[j][i];
			pos[1] = this->m_dataMatrix[j + 1][i];
			pos[2] = this->m_dataMatrix[j + 2][i];

			sample->SetLandmark(j / 3, pos);
		}

		Samples->push_back(sample);
	}

	std::vector< vtkSmartPointer< vtkPolyData >> backPolys;
	for (int i = 0; i < this->m_numberOfSamples; i++)
	{
		vtkSmartPointer< vtkPolyData > poly = vtkSmartPointer< vtkPolyData >::New();
		poly->DeepCopy(this->m_dataSetPoly.at(i));
		backPolys.push_back(poly);
	}

	for (int i = 0; i < this->m_numberOfSamples; i++)
	{
		pdm::Sample* sample = new pdm::Sample(Samples->at(i)->GetNumberOfLandmarks(), 3);
		sample = Samples->at(i);

		for (int j = 0; j < sample->GetNumberOfLandmarks(); j++)
		{
			pdm::VectorOfDouble pos(3); pos.fill(0.0);
			sample->GetLandmark(j, pos);

			double coords[3];
			coords[0] = pos[0]; coords[1] = pos[1]; coords[2] = pos[2];

			backPolys.at(i)->GetPoints()->SetPoint(j, coords);
		}
	}

	for (int i = 0; i < backPolys.size(); i++)
	{
		vtkSmartPointer< vtkTransformPolyDataFilter > transformFilter = vtkSmartPointer< vtkTransformPolyDataFilter >::New();
		transformFilter->SetInputData(backPolys.at(i));
		transformFilter->SetTransform(m_landmarkTransformVector.at(i));
		transformFilter->Update();
		backPolys.at(i)->SetPoints(transformFilter->GetOutput()->GetPoints());
	}

	return backPolys;
}


//=====================================================================================================


inline void RobustKPCA::eigenDecomposition(MatrixOfDouble _dataMatrix, MatrixOfDouble & _eigenVectors, VectorOfDouble & _eigenValues)
{
	int m = _dataMatrix.rows(); 
	int n = _dataMatrix.cols(); 

	VectorOfDouble mean(m, 0.0); 
	for (int i = 0; i < n; i++)
		mean += _dataMatrix.get_column(i); 
	mean /= n; 

	MatrixOfDouble covarianceMatrix(n, n, 0.0);

	for (int row = 0; row < n; row++)
	{
		for (int col = 0; col <= row; col++)
		{
			for (int cnt = 0; cnt < m; cnt++)
			{
				covarianceMatrix[row][col] += (_dataMatrix[cnt][row] - mean[cnt]) * (_dataMatrix[cnt][col] - mean[cnt]); 
			}

//			covarianceMatrix[row][col] /= m; 

			covarianceMatrix[col][row] = covarianceMatrix[row][col]; 
		}
	}

	vnl_symmetric_eigensystem<double> eigenSystem(covarianceMatrix);

	_eigenVectors = eigenSystem.V;
	_eigenValues = (eigenSystem.D).diagonal();
	_eigenVectors.fliplr();
	_eigenValues.flip();
}


inline void RobustKPCA::singularValueDecomposition(MatrixOfDouble _dataMatrix, MatrixOfDouble & _leftColumns, MatrixOfDouble & _svds, MatrixOfDouble & _rightColumns)
{
	vnl_svd< double > svd(_dataMatrix);   // _leftColumn * _rightColumn = I 

	_leftColumns = svd.U(); 
	_svds = svd.W(); 
	_rightColumns = svd.V().transpose(); 

}



/*******************************************************************************************/

// Important back up (The whole function)

//void RobustKPCA::RobustKernelNew(double gamma)
//{
//	//=======================================================================================
//	/* Perform Kernel PCA : Get K */
//
//	double n = this->m_numberOfSamples;
//
//	if (!m_kernelPCA) m_kernelPCA = new KernelPCA();
//
//	m_kernelPCA->setDataMatrix(this->m_dataMatrix);
//	m_kernelPCA->setGamma(gamma);
//	m_kernelPCA->RunKPCALearn();
//
//	MatrixOfDouble KernelMatrix(n, n, 0);
//	KernelMatrix = m_kernelPCA->getKMatrix();
//
//	vnl_svd< double > svdK(KernelMatrix);
//	double norm2 = svdK.sigma_max();
//
//	MatrixXd kernelM = MatrixXd::Zero(n, n);
//	for (int i = 0; i < n; i++)
//	{
//		for (int j = 0; j < n; j++)
//			kernelM(i, j) = KernelMatrix[i][j];
//	}
//
//	// SVD Kernel Matrix 
//	Eigen::JacobiSVD< MatrixXd > svd(kernelM, ComputeThinU | ComputeThinV);
//
//	VectorXd singularValues = svd.singularValues();
//	MatrixXd leftVectorsK = svd.matrixU();
//	MatrixXd rightVectorsK = svd.matrixV();
//	MatrixXd sigmasK = MatrixXd::Zero(n, n);
//	MatrixXd sqrtSigmasK = MatrixXd::Zero(n, n);
//	for (int i = 0; i < n; i++)
//	{
//		sigmasK(i, i) = singularValues(i);
//		sqrtSigmasK(i, i) = sqrt(singularValues(i));
//	}
//
//	MatrixXd WVT = MatrixXd::Zero(n, n);
//	WVT = sqrtSigmasK * rightVectorsK.transpose();
//
//	//==========================================================================================
//	/* Optimization via ALM */
//
//	double lambda = 1.0 / sqrt((double)std::fmax(this->m_numberOfLandmarks, n));  // lambda only influence the sparse matrix computation 
//	double tol1 = 1e-4;
//	double tol2 = 1e-7;
//	int maxIter = 500;
//	double epsilon = 0.1;  // balance the zeros 
//
//	/* Initialize L , Y = 0, P = I */
//
//	MatrixXd m_L = MatrixXd::Zero(n, n);
//	MatrixXd m_Y = MatrixXd::Zero(n, n);
//	MatrixXd m_E = MatrixXd::Zero(n, n);
//	MatrixXd m_I = MatrixXd::Identity(n, n); // need test 
//	MatrixXd sqrtXTX = MatrixXd::Identity(n, n);
//
//	double mu = 0.1 * norm2;  std::cout << "Initial mu = " << mu << std::endl;
//	double mu_max = mu * 1e6;
//	double rho = 1.1;
//
//	int iter = 0;
//	double converged = false;
//
//	MatrixXd _XTX = MatrixXd::Zero(n, n);
//	_XTX = 0.1 * kernelM;  // lambda * X^T * X 
//
//	/*MatrixOfDouble xtx(n, n, 0);
//	xtx = 0.1 * this->m_dataMatrix.transpose() * this->m_dataMatrix;
//	for (int i = 0; i < n; i++)
//	{
//	for (int j = 0; j < n; j++)
//	_XTX(i, j) = xtx[i][j];
//	}*/
//
//	MatrixXd m_L_old = MatrixXd::Zero(n, n);
//
//	// optimization via IRLS 
//	while (!converged)
//	{
//		iter++;
//
//		////////////////////////////////////////////////////////////////////////////
//		/* Get m_L (Z) */
//
//		MatrixXd tempA = /*2 * lambda **/ WVT.inverse() * kernelM;
//		arma::mat SylvesterA = zeros<mat>(n, n);
//		for (int i = 0; i < n; i++)
//		{
//			for (int j = 0; j < n; j++)
//				SylvesterA(i, j) = tempA(i, j);
//		}
//
//		MatrixXd tempB = WVT * sqrtXTX;
//		arma::mat SylvesterB = zeros<mat>(n, n);
//		for (int i = 0; i < n; i++)
//		{
//			for (int j = 0; j < n; j++)
//				SylvesterB(i, j) = tempB(i, j);
//		}
//
//		MatrixXd tempC = /*(-2) * lambda*/ WVT.inverse() * kernelM;
//		arma::mat SylvesterC = zeros<mat>(n, n);
//		for (int i = 0; i < n; i++)
//		{
//			for (int j = 0; j < n; j++)
//				SylvesterC(i, j) = -tempC(i, j);
//		}
//
//		arma::mat X = zeros<mat>(n, n);
//		X = arma::syl(SylvesterA, SylvesterB, SylvesterC);
//
//		// convert to L 
//		for (int i = 0; i < n; i++)
//		{
//			for (int j = 0; j < n; j++)
//				m_L(i, j) = X(i, j);
//		}
//
//		std::cout << std::endl;
//		std::cout << "m_L " << std::endl;
//		for (int i = 0; i < m_L.rows(); i++)
//		{
//			for (int j = 0; j < m_L.cols(); j++)
//				std::cout << m_L(i, j) << " , ";
//			std::cout << std::endl;
//		}
//		std::cout << std::endl;
//
//		//////////////////////////////////////////////////////////////////////////////
//		/* Get sqrtXTX (W) */
//
//		MatrixXd Temp = m_L.transpose() * kernelM * m_L + mu * mu * m_I;
//
//		Eigen::EigenSolver< MatrixXd > eig(Temp);
//		MatrixXd eigVectors = eig.eigenvectors().real();
//		VectorXd eigValues = eig.eigenvalues().real();
//		MatrixXd eigVectorsInverse = eig.eigenvectors().real().inverse();
//
//		MatrixXd sqrtEigValues = MatrixXd::Zero(n, n);
//		for (int i = 0; i < n; i++)
//		{
//			if (eigValues(i) >= 0)
//				sqrtEigValues(i, i) = sqrt(eigValues(i));
//		}
//
//		sqrtXTX = eigVectors * sqrtEigValues * eigVectorsInverse;
//		sqrtXTX = sqrtXTX.inverse();
//
//		////////////////////////////////////////////////////////////////////////
//
//		/*m_E = m_I - m_L;
//
//		MatrixXd W2 = MatrixXd::Zero(n, n);
//
//		for (int i = 0; i < n; i++)
//		{
//		W2(i, i) = m_E.col(i).transpose() * kernelM * m_E.col(i);
//		W2(i, i) += mu * mu;
//		}
//
//		sqrtXTX = sqrtXTX * W2; */
//
//		//////////////////////////////////////////////////////////////////////////////
//
//		mu = mu / rho;
//
//		MatrixOfDouble deltaL(n, n, 0);
//		for (int i = 0; i < n; i++)
//		{
//			for (int j = 0; j < n; j++)
//				deltaL[i][j] = m_L_old(i, j) - m_L(i, j);
//		}
//
//		MatrixOfDouble L(n, n, 0);
//		for (int i = 0; i < n; i++)
//		{
//			for (int j = 0; j < n; j++)
//				L[i][j] = m_L(i, j);
//		}
//
//		double stop = deltaL.frobenius_norm() / L.frobenius_norm();
//		if (stop < tol1)
//		{
//			converged = true;
//			break;
//		}
//
//		if (iter > 21)
//		{
//			std::cout << "Maximum iteration has reached !" << std::endl;
//			break;
//		}
//
//		std::cout << "tol " << iter << " : " << stop << std::endl;
//
//		m_L_old = m_L;
//
//	}
//
//	//while (iter < 20)
//	//{
//	//	iter++;
//
//	//	/**********************************************************************************/
//	//	/* Find m_L */
//
//	//	/* Solving L via gradient */
//
//	//	MatrixXd deltaL = m_I - m_E + (1.0 / mu) * m_Y; 
//
//	//	/* Sylvester Equation: AX + XB + C = 0*/
//	//	MatrixXd A = WVT.inverse(); 
//	//	MatrixXd B = (mu * sqrtXTX).inverse(); 
//
//	//	// convert to mat 
//	//	arma::mat SylvesterA = zeros<mat>(n, n); 
//	//	for (int i = 0; i < n; i++)
//	//	{
//	//		for (int j = 0; j < n; j++)
//	//			SylvesterA(i, j) = A(i, j); 
//	//	}
//
//	//	arma::mat SylvesterB = zeros<mat>(n, n); 
//	//	for (int i = 0; i < n; i++)
//	//	{
//	//		for (int j = 0; j < n; j++)
//	//			SylvesterB(i, j) = B(i, j); 
//	//	}
//
//	//	arma::mat SylvesterC = zeros<mat>(n, n); 
//	//	for (int i = 0; i < n; i++)
//	//	{
//	//		for (int j = 0; j < n; j++)
//	//			SylvesterC(i, j) = -deltaL(i, j); // -C 
//	//	}
//
//	//	arma::mat X = zeros<mat>(n, n); 
//	//	X = arma::syl(SylvesterA, SylvesterB, SylvesterC); 
//
//	//
//	//	// convert to L 
//	//	for (int i = 0; i < n; i++)
//	//	{
//	//		for (int j = 0; j < n; j++)
//	//			m_L(i, j) = X(i, j); 
//	//	}
//
//	//	m_L = WVT.inverse() * m_L; 
//
//	//	std::cout << "m_L " << iter << std::endl;
//	//	for (int i = 0; i < m_L.rows(); i++)
//	//	{
//	//		for (int j = 0; j < m_L.cols(); j++)
//	//			std::cout << m_L(i, j) << " , ";
//	//		std::cout << std::endl;
//	//	}
//	//	std::cout << std::endl;
//
//	//	/****************************************************************************/
//	//	/* Find SqrtXX */
//
//	//	MatrixXd XTX = m_L.transpose() * kernelM * m_L + epsilon * m_I;
//
//	//	// test print XTX
//	//	std::cout << "X^T X = L^T K L : " << std::endl;
//	//	for (int i = 0; i < n; i++)
//	//	{
//	//		for (int j = 0; j < n; j++)
//	//			std::cout << XTX(i, j) << ","; 
//	//		std::cout << std::endl;
//	//	}
//	//	std::cout << std::endl;
//
//	//	Eigen::EigenSolver< MatrixXd > eig(XTX);
//	//	MatrixXd eigVectors = eig.eigenvectors().real();
//	//	VectorXd eigValues = eig.eigenvalues().real();
//	//	MatrixXd eigVectorsInverse = eig.eigenvectors().real().inverse(); 
//
//	//	// test print
//	//	std::cout << "print eigenValues " << iter << std::endl;
//	//	for (int i = 0; i < n; i++)
//	//		std::cout << eigValues(i) << " , "; 
//	//	std::cout << std::endl;
//
//	//	// test print
//	//	std::cout << "print eigenVectors " << iter << std::endl;
//	//	for (int i = 0; i < n; i++)
//	//	{
//	//		for (int j = 0; j < n; j++)
//	//			std::cout << eigVectors(i, j) << ","; 
//	//		std::cout << std::endl;
//	//	}
//	//	std::cout << std::endl;
//
//	//	// test print
//	//	std::cout << "print eigenVectors inverse matrix " << iter << std::endl;
//	//	for (int i = 0; i < n; i++)
//	//	{
//	//		for (int j = 0; j < n; j++)
//	//			std::cout << eigVectorsInverse(i, j) << ",";
//	//		std::cout << std::endl;
//	//	}
//	//	std::cout << std::endl;
//
//	//	MatrixXd sqrtEigValues = MatrixXd::Zero(n, n); 
//	//	for (int i = 0; i < n; i++)
//	//	{
//	//		if (eigValues(i) >= 0)
//	//			sqrtEigValues(i, i) = sqrt(eigValues(i));
//	//	}
//
//	//	sqrtXTX = eigVectors * sqrtEigValues * (eigVectors.inverse());  
//
//	//	//arma::mat trans = arma::zeros(n, n); 
//	//	//for (int i = 0; i < n; i++)
//	//	//{
//	//	//	for (int j = 0; j < n; j++)
//	//	//		trans(i, j) = XTX(i, j); 
//	//	//}
//
//	//	//arma::mat sqrtTemp = arma::zeros(n, n); 
//
//	//	//sqrtTemp = sqrtmat(trans);
//	//	//sqrtTemp = sqrtmat_sympd(trans); // square root of symmetrix positive definite matrix 
//
//	//	//for (int i = 0; i < n; i++)
//	//	//{
//	//	//	for (int j = 0; j < n; j++)
//	//	//		sqrtXTX(i, j) = sqrtTemp(i, j); 
//	//	//}
//
//	//	// test print 
//	//	std::cout << "sqrtXTX : " << iter << std::endl;
//	//	for (int i = 0; i < n; i++)
//	//	{
//	//		for (int j = 0; j < n; j++)
//	//			std::cout << sqrtXTX(i, j) << ","; 
//	//		std::cout << std::endl; 
//	//	}
//	//	std::cout << std::endl; 
//
//	//	/******************************************************************************************/
//	//	/* Find m_E */
//	//	MatrixXd deltaE = m_I - m_L + (1.0 / mu) * m_Y; 
//
//	//	MatrixXd inverseE = MatrixXd::Zero(n, n); 
//
//	//	inverseE = (2 * lambda / mu) * kernelM + m_I; // (2KE + muE) is an identify matrix 
//	//	inverseE = inverseE.inverse(); 
//
//	//	m_E = inverseE * deltaE; 
//
//	//	/**********************************************************************************/
//
//	//	/* Stop criterion analysis */
//	//	MatrixXd Z = m_I - m_L - m_E; 
//
//	//	m_Y = m_Y + mu * Z;
//	//	mu = std::fmin(mu / rho, mu_max);
//	//
//	//	double stop1 = 0.0; 
//	//	/*double stop2 = 0;
//	//	for (int i = 0; i < n; i++)
//	//		stop2 += temp2[i][i];
//	//	stop2 = sqrt(stop2) / this->m_dataMatrix.frobenius_norm();*/
//
//	//	double stop2 = 0.0; 
//	///*	stop2 = temp2.frobenius_norm() / this->m_dataMatrix.frobenius_norm(); 
//
//	//	std::cout << "stop1 = " << stop1 << ", stop2 = " << stop2 << std::endl; std::cout << "svp = " << svp << std::endl;*/
//
//	//	if (stop1 < tol1 && stop2 < tol2)
//	//	{
//	//		converged = true;
//
//	//		std::cout << "Both stop criterions have been reached !" << std::endl; 
//	//		std::cout << "Iteration: " << iter << ", mu = " << mu << std::endl;
//	//	}
//
//	//	if (!converged && iter >= maxIter)
//	//	{
//	//		std::cout << "Maximum iterations reached" << std::endl;
//	//		converged = true;
//
//	//		std::cout << "Iteration: " << iter << ", mu = " << mu << "stop1 = " << stop1 << ", stop2 = " << stop2 << std::endl;
//	//	}
//
//	//	if (iter % 20 == 0)
//	//	{
//	//		std::cout << "iter = " << iter << ": stop1 = " << stop1 << ", stop2 = " << stop2 << ", mu = " << mu << std::endl;
//	//	}
//
//	//}// end while
//
//	//==========================================================================================
//	/* Return the low-rank subspace represented by L_{k+1} */
//
//	this->constructed_RKLRR.set_size(this->m_numberOfLandmarks, this->m_numberOfSamples);
//
//	MatrixOfDouble LTranspose(n, n, 0);
//	for (int i = 0; i < n; i++)
//	{
//		for (int j = 0; j < n; j++)
//			LTranspose[i][j] = m_L(i, j);
//	}
//	this->constructed_RKLRR = this->m_dataMatrix * LTranspose;
//
//	std::cout << "Robust Kernelized Low-Rank Representation Finished ... " << std::endl;
//
//}


/********************************************************************************************/

// do not delete 

/* Method : Robust Kernel PCA proposed in NIPS 2009
Robust Kernel PCA by robust the optimization approach
Optimization: arg min(z) ||phi(x) - phi(z)||^2 + C*||phi(z) - P(phi(x))||^2
arg min(z)              E_0(x,z) + C*E_proj(z)
@input : double _constant
@return: constructedM_NIPS2009

void RKPCA_NIPS2009(double _constant);


@return _output: the i_th reconstructed sample

RobustKPCA::VectorOfDouble RobustKPCA::Reconstruction_RKPCA_NIPS2009(vnl_vector< double > input, vnl_vector< double > output, double constant)
{
//MatrixOfDouble kernelMatrix(this->m_numberOfSamples, this->m_numberOfSamples);
//kernelMatrix = m_kernelPCA->getKMatrix();

//// m_coefficeints definition [k * n]
//MatrixOfDouble coefficients(m_kernelPCA->getEvecSelected().cols(), m_kernelPCA->getEvecSelected().rows());
//coefficients = m_kernelPCA->getEvecSelected().transpose();


//// meanK: [1 * n]
//VectorOfDouble meanK;  meanK.set_size( kernelMatrix.rows());
//for (unsigned int i = 0; i < kernelMatrix.rows(); i++)
//	meanK[i] = kernelMatrix.get_row(i).mean();
//
//// gammas: [1 * k]
//VectorOfDouble gammas; gammas.set_size( coefficients.rows());
//gammas = meanK * m_coefficients.transpose() ;
//
//
//// optimization function
//VectorOfDouble iterDiffs; iterDiffs.set_size(250); iterDiffs.fill(0);
//VectorOfDouble objFunc; objFunc.set_size(250); objFunc.fill(0);
//
//for (unsigned iter = 1; iter < 200; iter++)
//{
//	double k_input_output = m_kernelPCA->getKernel(input, output);
//
//	// k_M_output [1 * n]
//	VectorOfDouble k_M_output = m_kernelPCA->getKernel(m_dataMatrix, output);

//	// A [1 * k]
//	VectorOfDouble A; A.set_size( coefficients.rows() );
//	A = m_coefficients * k_M_output
//
//	// B: [m * k]
//	MatrixOfDouble rep_k_M_output; rep_k_M_output.set_size(m_dataMatrix.rows(), k_M_output.size());
//	for (unsigned int row = 0; row < m_dataMatrix.rows(); row++)
//		rep_k_M_output.set_row(row, k_M_output);  // repeat the row vector k_M_output: [m * n]
//
//	MatrixOfDouble dotMultiply; dotMultiply.set_size(m_dataMatrix.rows(), m_dataMatrix.cols());
//	for (unsigned int i = 0; i < m_dataMatrix.rows(); i++)
//	{
//		for (unsigned int j = 0; j < m_dataMatrix.cols(); j++)
//			dotMultiply[i][j] = m_dataMatrix[i][j] * rep_k_M_output[i][j];
//	}
//
//	// m_coeff: [k * n]
//	MatrixOfDouble B; B.set_size(m_dataMatrix.rows(), coefficients.rows());
//	B = dotMultiply.operator*( coefficients.transpose() );
//
//
//	// numerator : [m * 1]
//	VectorOfDouble numerator; numerator.set_size(input.size());
//	numerator = input.operator*(k_input_output)+ constant * (
//		k_M_output.pre_multiply(m_dataMatrix).operator/(m_dataSetPoly.size())  // m_dataMatrix * k_M_output'/n: [m * 1]
//		+ A.post_multiply(B.transpose())   // B * A [m * 1]
//		- gammas.pre_multiply(B));       // B * gammas [m * 1]
//
//	// dominator
//	double gammasA = 0.0;
//	for (unsigned int i = 0; i < A.size(); i++)
//		gammasA += gammas[i] * A[i]; // gammas' * A
//
//	double denominator = k_input_output + constant * (
//		k_M_output.sum() / m_dataSetPoly.size()
//		+ A.squared_magnitude()    // A' * A
//		- gammasA);
//
//	iterDiffs[iter] = (output - (numerator.operator/(denominator))).squared_magnitude();
//
//	// construct output
//	output = numerator.operator/(denominator);
//
//	if (iterDiffs[iter] < 1e-15)
//	{
//		std::cout << "iteration: " << iter << " ; ";
//		std::cout << "iterDiffs = " << iterDiffs[iter] << std::endl;
//		break;
//	}
//
//	// V: [k * 1]
//	VectorOfDouble V; V.set_size( coefficients.rows() );
//	V = (k_M_output - meanK).pre_multiply( coefficients );
//
//	double err = -2 / m_dataMatrix.cols() * k_M_output.sum() - V.squared_magnitude();
//
//	objFunc[iter] = -2 * m_kernelPCA->getKernel(output, input) + constant * err;
//}
//
return output;
}

void RobustKPCA::RobustKernelNew(double gamma)
{


double n = this->m_numberOfSamples;

if (!m_kernelPCA) m_kernelPCA = new KernelPCA();

m_kernelPCA->setDataMatrix(this->m_dataMatrix);
m_kernelPCA->setGamma(gamma);
m_kernelPCA->RunKPCALearn();

MatrixOfDouble KernelMatrix(n, n, 0);
KernelMatrix = m_kernelPCA->getKMatrix();

MatrixXd kernelM = MatrixXd::Zero(n, n);
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
kernelM(i, j) = KernelMatrix[i][j];
}

// SVD Kernel Matrix
Eigen::JacobiSVD< MatrixXd > svd(kernelM, ComputeThinU | ComputeThinV);

VectorXd singularValues = svd.singularValues();
MatrixXd leftVectorsK = svd.matrixU();
MatrixXd rightVectorsK = svd.matrixV();
MatrixXd sigmasK = MatrixXd::Zero(n, n);
MatrixXd sqrtSigmasK = MatrixXd::Zero(n, n);
for (int i = 0; i < n; i++)
{
sigmasK(i, i) = singularValues(i);
sqrtSigmasK(i, i) = sqrt(singularValues(i));
}



MatrixOfDouble SigmaK(n, n, 0);
MatrixOfDouble SqrtSigmaK(n, n, 0);
MatrixOfDouble LeftK(n, n, 0);
MatrixOfDouble RightK(n, n, 0);
MatrixOfDouble S_inverse(n, n, 0);  // S^{-1}, need test, for E


vnl_svd< double > svdK(KernelMatrix); // svd = U W V^T

SigmaK = svdK.W();
LeftK = svdK.U();
RightK = svdK.V();

for (int i = 0; i < n; i++)
SqrtSigmaK[i][i] = std::sqrt(SigmaK[i][i]);

MatrixOfDouble WVT(n, n, 0);
WVT = SqrtSigmaK * RightK.transpose();

// test print
std::cout << "WVT: " << std::endl;
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
std::cout << WVT[i][j] << ",";
std::cout << std::endl;
}
std::cout << std::endl;

double numPostive = 0;
for (int i = 0; i < this->m_numberOfSamples; i++)
{
S_inverse[i][i] = 1.0 / SqrtSigmaK[i][i];
if (S_inverse[i][i] > 0) numPostive++;
}

vnl_svd< double > svdM(this->m_dataMatrix);
double norm2 = svdM.sigma_max();

//==========================================================================================


double lambda = 1.0 / sqrt((double)std::fmax(this->m_numberOfLandmarks, n));  // lambda only influence the sparse matrix computation
double tol1 = 1e-4;
double tol2 = 1e-7;
int maxIter = 500;



MatrixOfDouble m_L(n, n, 0);
MatrixOfDouble m_Y(n, n, 0);
MatrixOfDouble m_E(n, n, 0);
MatrixOfDouble m_I(n, n, 0);
MatrixOfDouble sqrtXX(n, n, 0);  // A

for (int i = 0; i < n; i++)
{
m_I[i][i] = 1;
//		m_E[i][i] = 1;
sqrtXX[i][i] = 1;
}

double mu = 0.5;
double mu_max = mu * 1e6;
double rho = 1.6;

int iter = 0;
double converged = false;

while (iter < 20)
{
iter++;


int svp = 0;

MatrixOfDouble tempL = m_I - m_E + (1.0 / mu) * m_Y;

MatrixXd dataPoints = MatrixXd::Zero(n, n);

for (int i = 0; i < tempL.rows(); i++)
{
for (int j = 0; j < tempL.cols(); j++)
dataPoints(i, j) = tempL[i][j];
}

// Get mu A W V^T
MatrixOfDouble fracUp(n, n, 0);
fracUp = mu * sqrtXX * WVT;

// test print
std::cout << "sqrtXX: " << std::endl;
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
std::cout << sqrtXX[i][j] << ",";
std::cout << std::endl;
}
std::cout << std::endl;

// test print
std::cout << "FracUp: " << std::endl;
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
std::cout << fracUp[i][j] << ",";
std::cout << std::endl;
}
std::cout << std::endl;

MatrixOfDouble fracDown(n, n, 0);
fracDown = WVT + mu * sqrtXX;

// test print
std::cout << "fracDown before inverse: " << std::endl;
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
std::cout << fracDown[i][j] << ",";
std::cout << std::endl;
}
std::cout << std::endl;

MatrixXd inverseDown = MatrixXd::Zero(n, n);
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
inverseDown(i, j) = fracDown[i][j];
}
inverseDown = inverseDown.inverse();

MatrixOfDouble fracDownInversed(n, n, 0);
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
fracDownInversed[i][j] = inverseDown(i, j);
}

// test print
std::cout << "fracDown after inverse: " << std::endl;
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
std::cout << fracDownInversed[i][j] << ",";
std::cout << std::endl;
}
std::cout << std::endl;

// divide fracUp / fracDown
MatrixOfDouble frac(n, n, 0);
frac = fracUp * fracDownInversed;

// test print
std::cout << "frac: " << std::endl;
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
std::cout << frac[i][j] << ",";
std::cout << std::endl;
}
std::cout << std::endl;

m_L = frac * tempL;

std::cout << "m_L " << iter << std::endl;
for (int i = 0; i < m_L.rows(); i++)
{
for (int j = 0; j < m_L.cols(); j++)
std::cout << m_L[i][j] << " , ";
std::cout << std::endl;
}
std::cout << std::endl;


MatrixOfDouble XXT(n, n, 0);
XXT = m_L.transpose() * KernelMatrix * m_L;

// test print
std::cout << "XXT: " << iter << std::endl;
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
std::cout << XXT[i][j] << ",";
std::cout << std::endl;
}
std::cout << std::endl;

// eig(XXT)
MatrixXd EigXXT = MatrixXd::Zero(n, n);
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
EigXXT(i, j) = XXT[i][j];
}


Eigen::EigenSolver< MatrixXd > eigen(EigXXT);
MatrixXd eVec = eigen.eigenvectors().real();
VectorXd eVal = eigen.eigenvalues().real();

// test print
std::cout << "eVec: " << std::endl;
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
std::cout << eVec(i, j) << ",";
std::cout << std::endl;
}
std::cout << std::endl;

MatrixXd diagEVals = MatrixXd::Zero(n, n);
for (int i = 0; i < n; i++)
diagEVals(i, i) = sqrt(eVal(i));

EigXXT = eVec * diagEVals * eVec.inverse();  // test is required

for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
sqrtXX[i][j] = EigXXT(i, j);
}


//Eigen::JacobiSVD< MatrixXd > svd(dataPoints, ComputeThinU | ComputeThinV);

//VectorXd singularValues = svd.singularValues();
//MatrixXd leftVectors = svd.matrixU();
//MatrixXd rightVectors = svd.matrixV();

//// convert back
//VectorOfDouble diagSVec(singularValues.size(), 0);
//for (int i = 0; i < singularValues.size(); i++)
//	diagSVec[i] = singularValues(i);

//for (int i = 0; i < diagSVec.size(); i++)
//{
//	if (diagSVec[i] >(1.0 / mu))  // 1 / mu = 2, is too large
//		svp++;
//}

//MatrixOfDouble U(leftVectors.rows(), leftVectors.cols(), 0);
//MatrixOfDouble V(rightVectors.rows(), rightVectors.cols(), 0);

//for (int i = 0; i < leftVectors.rows(); i++)
//{
//	for (int j = 0; j < leftVectors.cols(); j++)
//		U[i][j] = leftVectors(i, j);
//}

//for (int i = 0; i < rightVectors.rows(); i++)
//{
//	for (int j = 0; j < rightVectors.cols(); j++)
//		V[i][j] = rightVectors(i, j);
//}

//MatrixOfDouble tempU(U.rows(), svp); tempU.fill(0.0);
//for (int u = 0; u < svp; u++)
//	tempU.set_column(u, U.get_column(u));

//MatrixOfDouble tempVTrans(V.rows(), svp); tempVTrans.fill(0.0);
//for (int v = 0; v < svp; v++)
//	tempVTrans.set_column(v, V.get_column(v));
//tempVTrans = tempVTrans.transpose();

//MatrixOfDouble tempW(svp, svp); tempW.fill(0.0);
//for (int w = 0; w < svp; w++)
//	tempW[w][w] = diagSVec[w] - 1.0 / mu;

//m_L = tempU * tempW * tempVTrans;

MatrixOfDouble m_H(n, n, 0);
m_H = m_I - m_L + (1.0 / mu) * m_Y;

double tau = mu / lambda;

for (int colP = 0; colP < n; colP++)
{
VectorOfDouble pk(n, 0);  pk = m_E.get_column(colP);
VectorOfDouble hk(n, 0);  hk = m_H.get_column(colP);

VectorOfDouble hu(numPostive, 0);
for (int i = 0; i < numPostive; i++)
hu[i] = hk[i];

VectorOfDouble hd(n - numPostive, 0);
for (int i = 0; i < n - numPostive; i++)
hd[i] = hk[i + numPostive];

double tolP = 0.0;
for (int i = 0; i < numPostive; i++)
tolP += std::pow(S_inverse[i][i] * hu[i], 2);
tolP = sqrt(tolP);

double alpha = 0.0;
for (int i = 0; i < n; i++)
alpha += std::pow(SqrtSigmaK[i][i] * pk[i], 2);
alpha = sqrt(alpha);

if (tolP > 1.0 / tau)
{
for (int row = 0; row < numPostive; row++)
m_E[row][colP] = (tau * alpha) / (tau * alpha + SigmaK[row][row]) * hu[row];

for (int row = numPostive; row < n; row++)
m_E[row][colP] = hd[row - numPostive];

m_E.set_column(colP, LeftK * m_E.get_column(colP));
}
else
{
for (int row = 0; row < numPostive; row++)
m_E[row][colP] = 0;

for (int row = numPostive; row < n; row++)
m_E[row][colP] = hd[row - numPostive];

m_E.set_column(colP, LeftK * m_E.get_column(colP));
}
}




MatrixOfDouble Z = m_I - m_L - m_E;

m_Y = m_Y + mu * Z;
mu = std::fmin(mu * rho, mu_max);
double stop1 = Z.operator_inf_norm();

MatrixOfDouble temp2 = Z.transpose() * KernelMatrix * Z;   // (X - XL - E).Fro()
//double stop2 = 0;
//for (int i = 0; i < n; i++)
//stop2 += temp2[i][i];
//stop2 = sqrt(stop2) / this->m_dataMatrix.frobenius_norm();

double stop2 = 0.0;
stop2 = temp2.frobenius_norm() / this->m_dataMatrix.frobenius_norm();

std::cout << "stop1 = " << stop1 << ", stop2 = " << stop2 << std::endl; std::cout << "svp = " << svp << std::endl;

if (stop1 < tol1 && stop2 < tol2)
{
converged = true;

std::cout << "Both stop criterions have been reached !" << std::endl;
std::cout << "Iteration: " << iter << ", svp = " << svp << ", mu = " << mu << std::endl;
}

if (!converged && iter >= maxIter)
{
std::cout << "Maximum iterations reached" << std::endl;
converged = true;

std::cout << "Iteration: " << iter << ", svp = " << svp << ", mu = " << mu << "stop1 = " << stop1 << ", stop2 = " << stop2 << std::endl;
}

if (iter % 20 == 0)
{
std::cout << "iter = " << iter << ": stop1 = " << stop1 << ", stop2 = " << stop2 << ", mu = " << mu << std::endl;
}

}// end while

//==========================================================================================


//VectorOfDouble norms(n, 0); std::cout << "norms = ";
//for (int i = 0; i < m_L.cols(); i++)
//{
//	norms[i] = std::sqrt( m_L.get_column(i).squared_magnitude() );
//	std::cout << norms[i] << " , ";
//}
//std::cout << std::endl;

//for (int j = 0; j < m_L.cols(); j++)
//	m_L.get_column(j).operator/(norms[j]);


// test
std::cout << std::endl;
std::cout << "m_L " << std::endl;
for (int i = 0; i < m_L.rows(); i++)
{
for (int j = 0; j < m_L.cols(); j++)
std::cout << m_L[i][j] << " , ";
std::cout << std::endl;
}
std::cout << std::endl;

this->constructed_RKLRR.set_size(this->m_numberOfLandmarks, this->m_numberOfSamples);
this->constructed_RKLRR = this->m_dataMatrix * m_L.transpose();

//// test print constructed RKLRR
//std::cout << " test print constructed RKLRR " << std::endl;
//for (int i = 0; i < 20; i++)
//{
//	for (int j = 0; j < n; j++)
//		std::cout << this->constructed_RKLRR[i][j] << " , ";
//	std::cout << std::endl;
//}
//std::cout << std::endl;


std::cout << "Robust Kernelized Low-Rank Representation Finished ... " << std::endl;

}


*/

/*********************************************************************************************/
// Back up KRPCA (decompose kernel matrix, for RKLRR)

// comment the whole function once 

//void RobustKPCA::RKPCA(double gamma)
//{
//	double n = this->m_numberOfSamples;
//
//	// Get initial kernel matrix 
//
//	MatrixOfDouble KMatrix(n, n, 0);
//	for (int a = 0; a < n; a++)
//	{
//		vnl_vector< double > Aa = this->m_dataMatrix.get_column(a);
//		for (int b = 0; b < n; b++)
//		{
//			vnl_vector< double > Bb = this->m_dataMatrix.get_column(b);
//
//			// Gaussian Kernel Matrix 
//			KMatrix[a][b] = std::exp(-gamma * (Aa - Bb).squared_magnitude());
//
//			KMatrix[b][a] = KMatrix[a][b];
//		}
//	}
//
//	// test print 
//	std::cout << std::endl;
//	std::cout << "print Initial kernel matrix .. " << std::endl;
//	for (int i = 0; i < n; i++)
//	{
//		for (int j = 0; j < n; j++)
//			std::cout << std::setprecision(6) << KMatrix[i][j] << ",";
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//
//	/**************************************************************************************/
//
//	double lambda = 0.5;
//	double tol = 1e-10;
//	int maxIter = 500;
//
//	MatrixOfDouble Y = KMatrix; // can be tuned
//
//	vnl_svd< double > svdY(Y);
//
//	double norm2 = svdY.sigma_max();
//	double normInf = Y.operator_inf_norm() / lambda;
//	double normDual = std::fmax(norm2, normInf);
//
//	Y = Y / normDual;
//
//	Y.fill(0.0);
//
//	MatrixOfDouble m_lowK(n, n, 0);  // return this 
//	MatrixOfDouble m_sparseK(n, n, 0);
//
//	double dNorm = KMatrix.frobenius_norm();
//
//	double mu = 1.25 /*0.5 / norm2*/;
//	double mu_bar = mu * 1e6;
//	double rho = 1.6;
//
//	int iter = 0;
//	int total_svd = 0;
//	double converged = false;
//	double stopCriterion = 1.0;
//
//	while (!converged)
//	{
//		iter++;
//
//		/**********************************************************************************/
//		vnl_svd< double > svd(KMatrix - m_sparseK + (1.0 / mu)* Y);
//		MatrixOfDouble diagS = svd.W();
//		VectorOfDouble diagSVec(diagS.rows());
//
//		for (int i = 0; i < diagS.rows(); i++)
//		{
//			diagSVec[i] = diagS[i][i];
//		}
//
//
//		int svp = 0;
//		for (int j = 0; j < diagSVec.size(); j++)
//		{
//			if (diagSVec[j] > 1 / mu)
//				svp++;
//		}
//
//		std::cout << "iter = " << iter << " , 1/mu = " << 1 / mu << std::endl;
//
//		// print
//		if (iter % 10 == 0 || iter == 1)
//		{
//			std::cout << "iteration " << iter << ": first 10 singular values are: ";
//			for (int j = 0; j < diagSVec.size(); j++)
//			{
//				if (j < 10) std::cout << diagSVec[j] << " , ";
//				else break;
//			}
//
//			std::cout << "; svp = " << svp << std::endl;
//		}
//
//
//		MatrixOfDouble U = svd.U();
//		MatrixOfDouble V = svd.V();
//		MatrixOfDouble tempU(U.rows(), svp); tempU.fill(0.0);
//		for (int u = 0; u < svp; u++)
//			tempU.set_column(u, U.get_column(u));
//
//		MatrixOfDouble tempVTrans(V.rows(), svp); tempVTrans.fill(0.0);
//		for (int v = 0; v < svp; v++)
//			tempVTrans.set_column(v, V.get_column(v));
//		tempVTrans = tempVTrans.transpose();
//
//		MatrixOfDouble tempW(svp, svp); tempW.fill(0.0);
//		for (int w = 0; w < svp; w++)
//		{
//			tempW[w][w] = diagSVec[w] - 1 / mu;
//		}
//
//		m_lowK = tempU * tempW * tempVTrans;
//		/**********************************************************************************/
//
//		/**********************************************************************************/
//		MatrixOfDouble temp_M = KMatrix - m_lowK + (1.0 / mu) * Y;
//
//		for (int eCol = 0; eCol < temp_M.cols(); eCol++)
//		{
//			for (int eRow = 0; eRow < temp_M.rows(); eRow++)
//			{
//				m_sparseK[eRow][eCol] = std::fmax(temp_M[eRow][eCol] - lambda / mu, 0.0)
//					+ std::fmin(temp_M[eRow][eCol] + lambda / mu, 0.0);
//			}
//		}
//		/**********************************************************************************/
//
//		total_svd++;
//
//		MatrixOfDouble Z = KMatrix - m_lowK - m_sparseK;
//
//		int test = 0;
//		for (int i = 0; i < Z.cols(); i++)
//		{
//			for (int j = 0; j < Z.rows(); j++)
//			{
//				if (abs(Z[j][i]) <= 0)
//					test++;
//			}
//		}
//		std::cout << "iter = " << iter << ", test = " << test << std::endl;
//
//		Y = Y + mu * Z;
//		mu = std::fmin(mu * rho, mu_bar);
//
//		stopCriterion = Z.frobenius_norm() / dNorm;
//
//		if (stopCriterion < tol)
//		{
//			converged = true;
//
//			int e = 0;
//			for (int eCol = 0; eCol < m_sparseK.cols(); eCol++)
//			{
//				for (int eRow = 0; eRow < m_sparseK.rows(); eRow++)
//				{
//					if (abs(m_sparseK[eRow][eCol]) > 0)
//						e++;
//				}
//			}
//
//			std::cout << "stopCriterion < tol : ";
//			std::cout << "#svd: " << total_svd << ", #entries in |sparse matrix|: " << e << ", stopCriterion: " << stopCriterion;
//			std::cout << " ; Iteration: " << iter << ", svp = " << svp << ", mu = " << mu << std::endl;
//		}
//
//		if ((total_svd % 10) == 0)
//		{
//			int e = 0;
//			for (int eCol = 0; eCol < m_sparseK.cols(); eCol++)
//			{
//				for (int eRow = 0; eRow < m_sparseK.rows(); eRow++)
//				{
//					if (abs(m_sparseK[eRow][eCol]) > 0)
//						e++;
//				}
//			}
//		}
//
//		if (!converged && iter >= maxIter)
//		{
//			std::cout << "Maximum iterations reached" << std::endl;
//			converged = true;
//		}
//
//	}// end while
//
//	if (!m_kernelPCA_KRPCASVT) m_kernelPCA_KRPCASVT = new KernelPCA();
//
//	m_kernelPCA_KRPCASVT->setDataMatrix(this->m_dataMatrix);m
//	m_kernelPCA_KRPCASVT->setGamma(gamma);
//
//	// Back project with the new Kernel Matrix 
//	m_kernelPCA_KRPCASVT->RunKPCALearnSpecifiedKMatrix(m_lowK);
//	m_kernelPCA_KRPCASVT->BackProjectInternal();
//
//	constructed_RKPCA.set_size(this->m_numberOfLandmarks, n); constructed_RKPCA.fill(0.0);
//	constructed_RKPCA = m_kernelPCA_KRPCASVT->getReconstructedMatrix();
//
//	// test print 
//	std::cout << "RKPCA constructed matrix ... " << std::endl;
//	for (int i = 0; i < 20; i++)
//	{
//		for (int j = 0; j < n; j++)
//		{
//			std::cout << this->constructed_RKPCA[i][j] << ",";
//		}
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//}


//******************************************************
// @input: datamatrix ; rankN ; lambda ; tol ; maxIter 
// 
// @output: low-rank Matrix ; sparse Matrix ; iter 
//******************************************************
//void RobustKPCA::PSSV_RPCA(int rankN)
//{
//	std::cout << std::endl;
//	std::cout << "Starting processing Robust PCA using PSSV ... " << std::endl;
//
//	if (rankN < 0)
//	{
//		rankN = 0;
//		std::cout << "rank N = 0 by default same as ALM.." << std::endl;
//	}
//
//	double m = m_dataMatrix.rows();
//	double n = m_dataMatrix.cols();
//
//	double lambda = 1.0 / sqrt((double)std::fmax(m, n)); // can be tuned 
//	double tol = 1e-10;   // can be tuned
//	int maxIter = 500;
//
//	MatrixOfDouble Y = m_dataMatrix; // initialization Lagrange multiplier, can be tuned
//	vnl_svd< double > svdY(Y);
//
//	double norm2 = svdY.sigma_max();
//	double normInf = Y.operator_inf_norm() / lambda;
//	double normDual = std::fmax(norm2, normInf);
//
//	Y = Y / normDual;
//
//	MatrixOfDouble m_lowM(m, n); m_lowM.fill(0.0); // initialization low-rank Matrix , can be tuned
//	MatrixOfDouble m_sparseM(m, n); m_sparseM.fill(0.0); // can be tuned 
//
//	double mu = 1.25 / norm2; // can be tuned
//	double mu_bar = mu * 1e7; // can be tuned 
//	double rho = 1.6;
//
//	double dNorm = m_dataMatrix.frobenius_norm();
//
//	int iter = 0;
//	int total_svd = 0;
//	double converged = false;
//	double stopCriterion = 1.0;
//
//	while (!converged)
//	{
//		iter++;
//
//		/********************************************************************/
//		// A = U * W * V 
//		vnl_svd< double > svd(m_dataMatrix - m_sparseM + (1.0 / mu)* Y);
//		MatrixOfDouble diagS = svd.W();
//
//		VectorOfDouble diagSVec(diagS.rows());
//		for (int i = 0; i < diagS.rows(); i++)
//		{
//			diagSVec[i] = diagS[i][i];
//		}
//
//		vnl_svd< double > svdM(m_dataMatrix);
//		MatrixOfDouble weightings = svdM.W();
//		VectorOfDouble diagWeighting(weightings.rows());
//
//		for (int i = 0; i < weightings.rows(); i++)
//			diagWeighting[i] = weightings[i][i];
//
//		// find the length of svp 
//		int svp = 0;
//		for (int j = 0; j < diagSVec.size(); j++)
//		{
//			if (diagSVec[j] >(1.0 / mu))
//				svp++;
//		}
//
//		// print
//		if (iter % 10 == 0 || iter == 1)
//		{
//			std::cout << "iteration " << iter << ": first 10 singular values are: ";
//			for (int j = 0; j < diagSVec.size(); j++)
//			{
//				if (j < 10) std::cout << diagSVec[j] << " , ";
//				else break;
//			}
//			std::cout << "with the sum is " << diagSVec.sum() << std::endl;
//
//			std::cout << "initial singulars = ";
//			for (int i = 0; i < diagWeighting.size(); i++)
//				std::cout << diagWeighting[i] << " ";
//			std::cout << ", sum = " << diagWeighting.sum() << std::endl;
//
//		}
//
//
//		MatrixOfDouble U = svd.U();
//		MatrixOfDouble V = svd.V();
//
//		MatrixOfDouble tempU(U.rows(), svp); tempU.fill(0.0);
//		for (int u = 0; u < svp; u++)
//			tempU.set_column(u, U.get_column(u));
//
//		MatrixOfDouble tempVTrans(V.rows(), svp); tempVTrans.fill(0.0);
//		for (int v = 0; v < svp; v++)
//			tempVTrans.set_column(v, V.get_column(v));
//		tempVTrans = tempVTrans.transpose();
//
//		MatrixOfDouble dterm(svp, svp); dterm.fill(0.0);
//
//		/*************************************/
//		if (svp <= rankN)  // w < rankN 
//		{
//			for (int w = 0; w < svp; w++)
//				dterm[w][w] = diagSVec[w];  // diagSVec[w] > 1.0 / mu
//		}
//		else // svp > rankN 
//		{
//			for (int w = 0; w < svp; w++)
//			{
//				if (w < rankN)
//					dterm[w][w] = diagSVec[w];   // 0 <= w < rankN 
//				else
//					dterm[w][w] = diagSVec[w] - 1.0 / mu;  // rankN <= w < svp 
//			}
//		}
//
//
//		m_lowM = tempU * dterm * tempVTrans;
//
//		/*****************************************************************/
//
//		MatrixOfDouble temp_M = m_dataMatrix - m_lowM + (1 / mu)* Y;
//
//		for (int sCol = 0; sCol < m_dataMatrix.cols(); sCol++)
//		{
//			for (int sRow = 0; sRow < m_dataMatrix.rows(); sRow++)
//			{
//				m_sparseM[sRow][sCol] = std::fmax(temp_M[sRow][sCol] - lambda / mu, 0)
//					+ std::fmin(temp_M[sRow][sCol] + lambda / mu, 0);
//			}
//		}
//
//		/********************************************************************/
//
//		total_svd++;
//
//		MatrixOfDouble Z = m_dataMatrix - m_lowM - m_sparseM;
//		Y = Y + mu * Z;
//		mu = std::fmin(mu * rho, mu_bar);
//
//		stopCriterion = Z.frobenius_norm() / dNorm;
//
//		if (stopCriterion < tol)
//		{
//			converged = true;
//
//			int e = 0;
//			for (int eCol = 0; eCol < m_sparseM.cols(); eCol++)
//			{
//				for (int eRow = 0; eRow < m_sparseM.rows(); eRow++)
//				{
//					if (abs(m_sparseM[eRow][eCol]) > 0)
//						e++;
//				}
//			}
//
//			std::cout << "stopCriterion < tol : ";
//			std::cout << "#svd: " << total_svd << ", #entries in |sparse matrix|: " << e << ", stopCriterion: " << stopCriterion;
//			std::cout << " ; Iteration: " << iter << ", svp = " << svp << ", mu = " << mu << std::endl;
//		}
//
//		if ((total_svd % 10) == 0)
//		{
//			int e = 0;
//			for (int eCol = 0; eCol < m_sparseM.cols(); eCol++)
//			{
//				for (int eRow = 0; eRow < m_sparseM.rows(); eRow++)
//				{
//					if (abs(m_sparseM[eRow][eCol]) > 0)
//						e++;
//				}
//			}
//
//			/*std::cout << "total_svd % 10 : ";
//			std::cout << "#svd: " << total_svd << ", #entries in |E|_0: " << e << ", stopCriterion: " << stopCriterion;*/
//			std::cout << "Iteration: " << iter << ", svp = " << svp << ", mu = " << mu << std::endl;
//		}
//
//
//		if (!converged && iter >= maxIter)
//		{
//			std::cout << "Maximum iterations reached" << std::endl;
//			converged = true;
//		}
//
//	}// end while 
//
//	this->constructed_PSSV_RPCA.set_size(m, n); this->constructed_PSSV_RPCA.fill(0.0);
//	this->constructed_PSSV_RPCA = m_lowM;
//
//}

//////////// DO NOT DELETE : ERROR 0.01 //////////////////////////////////////////////////////////////////////

//vnl_vector< double > gramVector(this->nSamples, 0.0); 
//for (int i = 0; i < gramVector.size(); i++)
//{
//	double dist = vnl_vector_ssd(Z, this->_dataMatrix.get_column(i)); 
//	gramVector[i] = exp(-dist / (2 * this->gamma * this->gamma)); 
//}

/////// **** lower **** /////

//double lower = 0.0; 
//for (int i = 0; i < gramVector.size(); i++)
//	lower += gramVector[i] * _gammas[i]; 

//if (lower == 0)
//{
//	cout << " sum = zero ! reset the kernel width sigma " << endl;
//	lower = 0.00001;   // avoid 0.0 
//}

/////// **** upper **** /////

//vnl_vector< double > kernelPointNew(this->nSamples, 0.0); 
//for (int i = 0; i < this->nSamples; i++)
//{
//	kernelPointNew[i] = _gammas[i] * gramVector[i]; 

//	Z += kernelPointNew[i] * this->_dataMatrix.get_column(i); 
//}

//Z /= lower; 

/////// **** tolerance **** /////

//if ((preImgZ - Z).two_norm() / Z.two_norm() < tol)
//{
//	cout << " iteration < tol = " << iter << endl;
//	return Z;
//}

//iter++;
/////////////////////////////////////////////////////////////////////////////////////////////


/////// **** KRLRR ***** ///// 

//void RobustKPCA::RobustKernelLRR()
//{
//	this->m_modelType = "RKLRR";
//
//	if (this->m_numberOfSamples == 0)
//	{
//		std::cout << "Please load the data polys !" << std::endl;
//		return;
//	}
//
//	unsigned int n = this->m_numberOfSamples;
//	this->m_gramMatrix.set_size(n, n);
//
//	for (int a = 0; a < this->m_numberOfSamples; a++)
//	{
//		VectorOfDouble Aa = this->m_dataMatrix.get_column(a);
//		for (int b = 0; b < this->m_numberOfSamples; b++)
//		{
//			VectorOfDouble Bb = this->m_dataMatrix.get_column(b);
//			double dist = vnl_vector_ssd(Aa, Bb);
//			this->m_gramMatrix[a][b] = exp(-dist / (2 * this->m_gamma * this->m_gamma));
//		}
//	}
//	MatrixXd kernelM = MatrixXd::Zero(n, n);
//	for (int i = 0; i < n; i++)
//	{
//		for (int j = 0; j < n; j++)
//			kernelM(i, j) = this->m_gramMatrix[i][j];
//	}
//
//	// SVD Kernel Matrix 
//	Eigen::JacobiSVD< MatrixXd > svd(kernelM, ComputeThinU | ComputeThinV);
//	VectorXd singularValues = svd.singularValues();
//	MatrixXd leftVectorsK = svd.matrixU();
//	MatrixXd rightVectorsK = svd.matrixV();
//	MatrixXd sigmasK = MatrixXd::Zero(n, n);
//	MatrixXd sqrtSigmasK = MatrixXd::Zero(n, n);
//	for (int i = 0; i < n; i++)
//	{
//		sigmasK(i, i) = singularValues(i);
//		sqrtSigmasK(i, i) = sqrt(singularValues(i));
//	}
//
//	MatrixOfDouble SigmaK(n, n, 0);
//	MatrixOfDouble SqrtSigmaK(n, n, 0);
//	MatrixOfDouble LeftK(n, n, 0);
//	MatrixOfDouble RightK(n, n, 0);
//	MatrixOfDouble S_inverse(n, n, 0);  // S^{-1}, need test, for E
//
//	vnl_svd< double > svdK(this->m_gramMatrix); // svd = U W V^T
//	SigmaK = svdK.W();
//	LeftK = svdK.U();
//	RightK = svdK.V();
//	for (int i = 0; i < n; i++)
//		SqrtSigmaK[i][i] = std::sqrt(SigmaK[i][i]);
//
//	double numPostive = 0;
//	for (int i = 0; i < this->m_numberOfSamples; i++)
//	{
//		S_inverse[i][i] = 1.0 / SqrtSigmaK[i][i];
//		if (S_inverse[i][i] > 0) numPostive++;
//	}
//
//	vnl_svd< double > svdM(this->m_dataMatrix);
//	double norm2 = svdM.sigma_max();
//
//	//==========================================================================================
//
//	double lambda = 1.0 / sqrt((double)std::fmax(this->m_numberOfLandmarks, n));  // lambda only influence the sparse matrix computation 
//	double tol1 = 1e-4;
//	double tol2 = 1e-7;
//	int maxIter = 500;
//	MatrixOfDouble m_L(n, n, 0);
//	MatrixOfDouble m_Y(n, n, 0);
//	MatrixOfDouble m_E(n, n, 0);
//	MatrixOfDouble m_I(n, n, 0);
//	for (int i = 0; i < n; i++)
//	{
//		m_I[i][i] = 1;
//		m_E[i][i] = 1;
//	}
//	double mu = 0.25;
//	double mu_max = mu * 1e6;
//	double rho = 1.1;
//	int iter = 0;
//	double converged = false;
//	while (!converged)
//	{
//		iter++;
//		int svp = 0;
//
//		MatrixOfDouble tempL = m_I - m_E + (1.0 / mu) * m_Y;
//		MatrixXd dataPoints = MatrixXd::Zero(n, n);
//		for (int i = 0; i < tempL.rows(); i++)
//		{
//			for (int j = 0; j < tempL.cols(); j++)
//				dataPoints(i, j) = tempL[i][j];
//		}
//
//		Eigen::JacobiSVD< MatrixXd > svd(dataPoints, ComputeThinU | ComputeThinV);
//		VectorXd singularValues = svd.singularValues();
//		MatrixXd leftVectors = svd.matrixU();
//		MatrixXd rightVectors = svd.matrixV();
//
//		// convert back 
//		VectorOfDouble diagSVec(singularValues.size(), 0);
//		for (int i = 0; i < singularValues.size(); i++)
//			diagSVec[i] = singularValues(i);
//		for (int i = 0; i < diagSVec.size(); i++)
//		{
//			if (diagSVec[i] >(1.0 / mu))  // 1 / mu = 2, is too large
//				svp++;
//		}
//
//		MatrixOfDouble U(leftVectors.rows(), leftVectors.cols(), 0);
//		MatrixOfDouble V(rightVectors.rows(), rightVectors.cols(), 0);
//		for (int i = 0; i < leftVectors.rows(); i++)
//		{
//			for (int j = 0; j < leftVectors.cols(); j++)
//				U[i][j] = leftVectors(i, j);
//		}
//		for (int i = 0; i < rightVectors.rows(); i++)
//		{
//			for (int j = 0; j < rightVectors.cols(); j++)
//				V[i][j] = rightVectors(i, j);
//		}
//
//		MatrixOfDouble tempU(U.rows(), svp, 0.0);
//		for (int u = 0; u < svp; u++)
//			tempU.set_column(u, U.get_column(u));
//
//		MatrixOfDouble tempVTrans(V.rows(), svp, 0.0);
//		for (int v = 0; v < svp; v++)
//			tempVTrans.set_column(v, V.get_column(v));
//
//		tempVTrans = tempVTrans.transpose();
//		MatrixOfDouble tempW(svp, svp, 0.0);
//		for (int w = 0; w < svp; w++)
//			tempW[w][w] = diagSVec[w] - 1.0 / mu;
//
//		m_L = tempU * tempW * tempVTrans;
//
//		//////////////////////////////////////////////////////////////////////////////////////
//
//		/* Get E */
//		MatrixOfDouble m_H(n, n, 0);
//		m_H = m_I - m_L + (1.0 / mu) * m_Y;
//		double tau = mu / lambda;
//		for (int colP = 0; colP < n; colP++)
//		{
//			VectorOfDouble pk(n, 0);  pk = m_E.get_column(colP);
//			VectorOfDouble hk(n, 0);  hk = m_H.get_column(colP);
//			VectorOfDouble hu(numPostive, 0);
//			for (int i = 0; i < numPostive; i++)
//				hu[i] = hk[i];
//			VectorOfDouble hd(n - numPostive, 0);
//			for (int i = 0; i < n - numPostive; i++)
//				hd[i] = hk[i + numPostive];
//			double tolP = 0.0;
//			for (int i = 0; i < numPostive; i++)
//				tolP += std::pow(S_inverse[i][i] * hu[i], 2);
//			tolP = sqrt(tolP);
//			double alpha = 0.0;
//			for (int i = 0; i < n; i++)
//				alpha += std::pow(SqrtSigmaK[i][i] * pk[i], 2);
//			alpha = sqrt(alpha);
//			if (tolP > 1.0 / tau)
//			{
//				for (int row = 0; row < numPostive; row++)
//					m_E[row][colP] = (tau * alpha) / (tau * alpha + SigmaK[row][row]) * hu[row];
//				for (int row = numPostive; row < n; row++)
//					m_E[row][colP] = hd[row - numPostive];
//				m_E.set_column(colP, LeftK * m_E.get_column(colP));
//			}
//			else
//			{
//				for (int row = 0; row < numPostive; row++)
//					m_E[row][colP] = 0;
//				for (int row = numPostive; row < n; row++)
//					m_E[row][colP] = hd[row - numPostive];
//				m_E.set_column(colP, LeftK * m_E.get_column(colP));
//			}
//		}
//
//		///////////////////////////////////////////////////////////////////////////////
//		MatrixOfDouble Z = m_I - m_L - m_E;
//		m_Y = m_Y + mu * Z;
//		mu = std::fmin(mu * rho, mu_max);
//		double stop1 = Z.operator_inf_norm();
//		MatrixOfDouble temp2 = Z.transpose() * this->m_gramMatrix * Z;   // (X - XL - E).Fro()  
//
//		//double stop2 = 0;
//		//for (int i = 0; i < n; i++)
//		//stop2 += temp2[i][i];
//		//stop2 = sqrt(stop2) / this->m_dataMatrix.frobenius_norm();
//
//		double stop2 = 0.0;
//		stop2 = temp2.frobenius_norm() / this->m_dataMatrix.frobenius_norm();
//		std::cout << "stop1 = " << stop1 << ", stop2 = " << stop2 << std::endl; std::cout << "svp = " << svp << std::endl;
//		if (stop1 < tol1 && stop2 < tol2)
//		{
//			converged = true;
//			std::cout << "Both stop criterions have been reached !" << std::endl;
//			std::cout << "Iteration: " << iter << ", svp = " << svp << ", mu = " << mu << std::endl;
//		}
//		if (!converged && iter >= maxIter)
//		{
//			std::cout << "Maximum iterations reached" << std::endl;
//			converged = true;
//			std::cout << "Iteration: " << iter << ", svp = " << svp << ", mu = " << mu << "stop1 = " << stop1 << ", stop2 = " << stop2 << std::endl;
//		}
//		if (iter % 20 == 0)
//		{
//			std::cout << "iter = " << iter << ": stop1 = " << stop1 << ", stop2 = " << stop2 << ", mu = " << mu << std::endl;
//		}
//
//	}
//
//	//=========================================================================================
//	// test 
//	std::cout << std::endl;
//	std::cout << "m_L in RKLRR " << std::endl;
//	for (int i = 0; i < m_L.rows(); i++)
//	{
//		for (int j = 0; j < m_L.cols(); j++)
//			std::cout << m_L[i][j] << " , ";
//		std::cout << std::endl;
//	}
//	std::cout << std::endl;
//	this->m_gramMatrix = m_L;
//
//
//	cout << __FUNCTION__ << " L: " << endl;
//	for (int i = 0; i < n; i++)
//	{
//		for (int j = 0; j < n; j++)
//		{
//			this->m_gramMatrix[i][j] = m_L(i, j);
//			cout << m_L(i, j) << " ";
//		}
//		cout << endl;
//	}
//	cout << endl;
//
//	///////////////////////////////////////////////////////////////////////////
//
//	this->constructed_RobustKernelLRR.set_size(this->m_dataMatrix.rows(), n);
//	this->constructed_RobustKernelLRR = this->m_dataMatrix * this->m_gramMatrix;
//
//
//	/** Center Matrix H **/
//	MatrixOfDouble identity = MatrixOfDouble(n, n, 0.0);
//	identity.set_identity();
//
//	/** centralizeKMatrix **/
//	MatrixOfDouble Ones = MatrixOfDouble(n, n, 1.0);
//	MatrixOfDouble H = identity - Ones / n;
//	MatrixOfDouble kCenteredMatrix = H * this->m_gramMatrix * H;
//
//	vnl_svd< double > svdKK(kCenteredMatrix);
//
//	this->m_eigenVectors = svdKK.U();
//	this->m_eigenVals = svdKK.W().diagonal();
//	double eps = 1e-04;  //5/2;
//	for (unsigned int i = 0; i < n; i++)
//	{
//		if (this->m_eigenVals[i] < eps)
//		{
//			this->m_eigenVals[i] = eps;
//			break;
//		}
//	}
//
//	// Normalize row such that for each eigenvector a with eigenvalue w we have w(a.a) = 1
//	for (unsigned int i = 0; i < this->m_eigenVectors.cols(); i++)
//	{
//		this->m_eigenVectors.scale_column(i, 1.0 / (sqrt(this->m_eigenVals[i])));
//	}
//	//	this->m_eigenVals /= static_cast<double>(n - 1);
//
//
//	/** Compute Kernel Principal Components **/
//	this->m_kernelPCs.set_size(n, n);
//	this->m_kernelPCs = this->m_gramMatrix * this->m_eigenVectors;
//
//
//	/** selected all eigenValues and eigenVectors **/
//	double cum = 0;
//	for (unsigned int i = 0; i < this->m_eigenVals.size(); i++)
//	{
//		cum += this->m_eigenVals.get(i);
//		if (cum / this->m_eigenVals.sum() >= 0.97)
//		{
//			this->m_numberOfRetainedPCs = i + 1; // count number
//			break;
//		}
//	}
//
//
//	/** extract selected all retainedPCs from eigenvectors and eigenvalues **/
//	this->m_eigenVectorsReduced.set_size(this->m_eigenVectors.rows(), this->m_numberOfRetainedPCs);
//	this->m_eigenVectorsReduced = this->m_eigenVectors.extract(this->m_eigenVectors.rows(), this->m_numberOfRetainedPCs, 0, 0);
//	this->m_eigenValsReduced.set_size(this->m_numberOfRetainedPCs);
//	this->m_eigenValsReduced = this->m_eigenVals.extract(this->m_numberOfRetainedPCs, 0);
//
//
//	/** Reduced Kernel Principal Components **/
//	this->m_reducedKernelPCs.set_size(n, this->m_numberOfRetainedPCs);
//	this->m_reducedKernelPCs = this->m_gramMatrix * this->m_eigenVectorsReduced;
//	this->constructed_RobustKernelLRR.set_size(this->m_dataMatrix.rows(), this->m_dataMatrix.cols());
//	for (int i = 0; i < n; i++)
//	{
//		VectorOfDouble output = this->backProject(this->m_reducedKernelPCs.get_row(i));  // k 
//		this->constructed_RobustKernelLRR.set_column(i, output);
//	}
//	if (constructed_RobustKernelLRR.cols() != n || constructed_RobustKernelLRR.rows() != m_numberOfLandmarks * 3)
//	{
//		std::cout << "dimensions (number of samples) do not match! " << std::endl;
//		return;
//	}
//	std::cout << " Finish performRobustKLRR in [RobustKPCA] -> Got constructed_RobustKPCA " << std::endl;
//
//}




