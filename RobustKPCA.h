#pragma once 

// ===========================================================================================
//   @ Creation Date:  Feb 2017 
//   @ Created  By  :  Jingting 
//
//   @ brief:  this class_RobustKPCA provides several methods imperfect shape reconstruction 
//             (1) Standard KPCA   
//             (2) Robust PCA solved via ALM             - ICML 2011 Canndes et al. 
//             (3) Kernelized Robust PCA                 - MICCAI 2017
//             (4) Robust Kernel PCA                     - Under Review of MedIA 
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
		constructed_RobustPCA.fill(0);
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
	 *  @brief   perform KPCA internal learn
	 *  @param   get the m_eigenVectors / m_eigenVals
	 *                   m_origGramMatrix / m_origKernelPCs
	 */
	void performKPCA();


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

	std::vector< vtkSmartPointer< vtkPolyData>> getConstructed_RobustPCA();

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

	MatrixOfDouble constructed_RobustPCA;         /**> Constructed Data Matrix of RobustPCA **/
	MatrixOfDouble constructed_KPCA;              /**> Constructed pre-images using KPCA **/
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
