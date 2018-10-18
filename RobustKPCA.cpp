#include "RobustKPCA.h" 

#include "armadillo_bits\fn_syl_lyap.hpp"


RobustKPCA::RobustKPCA()
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
//	constructed_RobustKernelPCA.fill(0);
	constructed_RobustPCA.fill(0);
	constructed_PCA.fill(0);
	constructed_RKPCA.fill(0); 
	constructed_Miccai17.fill(0); 

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

	this->m_gamma = 150;
	this->m_alignment = true; 
	this->m_scaling = false;

	this->m_meanShape = NULL;
	this->m_modelType = "";

}


RobustKPCA::~RobustKPCA()
{
	delete this->m_Utils;
}


void RobustKPCA::ReadDataMatrix(std::vector< vtkSmartPointer< vtkPolyData >> m_polyDataSet)
{
	if (!this->m_meanShape) this->m_meanShape = vtkSmartPointer< vtkPolyData >::New();

	this->m_inputMeanShapeVector = VectorOfDouble(m_polyDataSet.at(0)->GetNumberOfPoints() * 3, 0.0);

	for (int i = 0; i < m_polyDataSet.size(); i++)
	{
		VectorOfDouble vec = VectorOfDouble(m_polyDataSet.at(i)->GetNumberOfPoints() * 3, 0.0);
		for (int j = 0; j < m_polyDataSet.at(i)->GetNumberOfPoints() * 3; j += 3)
		{
			vec[j] = m_polyDataSet.at(i)->GetPoint(j / 3)[0];
			vec[j + 1] = m_polyDataSet.at(i)->GetPoint(j / 3)[1];
			vec[j + 2] = m_polyDataSet.at(i)->GetPoint(j / 3)[2];
		}

		this->m_inputMeanShapeVector += vec;
	}
	this->m_inputMeanShapeVector /= m_polyDataSet.size();

	// copy PolyDataSets 
	for (int i = 0; i < m_polyDataSet.size(); i++)
	{
		vtkSmartPointer< vtkPolyData > poly = vtkSmartPointer< vtkPolyData >::New();
		poly->DeepCopy(m_polyDataSet.at(i));
		m_dataSetPoly.push_back(poly);
	}

//	this->AlignDataSetsWithoutScaling(this->m_dataSetPoly);  the datasets input are supposed to be aligned 

	for (int i = 0; i < this->m_dataSetPoly.size(); i++)
	{
		vtkSmartPointer< vtkLandmarkTransform > landmarkTransform = vtkSmartPointer< vtkLandmarkTransform >::New();
		landmarkTransform->SetSourceLandmarks(this->m_dataSetPoly.at(i)->GetPoints());
		landmarkTransform->SetTargetLandmarks(m_polyDataSet.at(i)->GetPoints());
		landmarkTransform->SetModeToRigidBody();
		landmarkTransform->Update();
		m_landmarkTransformVector.push_back(landmarkTransform);
	}
	
	// *************************************************************************

	this->m_numberOfSamples = this->m_dataSetPoly.size();
	this->m_numberOfLandmarks = this->m_dataSetPoly.at(0)->GetNumberOfPoints();
	this->m_meanShapeVector = VectorOfDouble(this->m_numberOfLandmarks * 3, 0.0);

	unsigned int dimensions = this->m_dataSetPoly.at(0)->GetNumberOfPoints() * 3;

	this->m_dataMatrix.set_size(dimensions, m_numberOfSamples);
	this->m_dataMatrixInput.set_size(dimensions, m_numberOfSamples); 

	for (unsigned int i = 0; i < this->m_dataSetPoly.size(); i++)
	{
		vtkSmartPointer< vtkPolyData > poly = this->m_dataSetPoly.at(i);

		for (unsigned int j = 0; j < dimensions; j += 3)
		{
			double* pos = poly->GetPoint(j / 3);

			this->m_dataMatrix[j][i] = pos[0];
			this->m_dataMatrix[j + 1][i] = pos[1];
			this->m_dataMatrix[j + 2][i] = pos[2];

			this->m_dataMatrixInput[j][i] = pos[0];
			this->m_dataMatrixInput[j + 1][i] = pos[1];
			this->m_dataMatrixInput[j + 2][i] = pos[2];
		}
	}

	if (this->m_dataMatrix.cols() != this->m_dataSetPoly.size() || this->m_dataMatrix.rows() != this->m_dataSetPoly.at(0)->GetNumberOfPoints() * 3)
	{
		std::cout << "Wrong dimensions !" << std::endl;
		return;
	}

	if (this->m_dataMatrixInput.cols() != this->m_dataSetPoly.size() || this->m_dataMatrixInput.rows() != this->m_dataSetPoly.at(0)->GetNumberOfPoints() * 3)
	{
		std::cout << "Wrong dimensions !" << std::endl;
		return;
	}

	for (int i = 0; i < this->m_dataMatrix.cols(); i++)
		this->m_meanShapeVector += this->m_dataMatrix.get_column(i);
	this->m_meanShapeVector /= this->m_numberOfSamples;

	cout << __FUNCTION__ << " Done ... " << endl; 

}


/**********************************************************************************************/


void RobustKPCA::performKPCA()
{
	this->m_modelType = "KPCA"; 

	// utilize this->m_dataMatrix 
	if (this->m_numberOfSamples == 0)
	{
		std::cout << "Please load the data polys !" << std::endl;
		return;
	}

	cout << __FUNCTION__ << " kernel matrix " << endl; 
	this->m_gramMatrix.set_size(this->m_numberOfSamples, this->m_numberOfSamples);
	for (int a = 0; a < this->m_numberOfSamples; a++)
	{
		VectorOfDouble Aa = this->m_dataMatrix.get_column(a);
		for (int b = 0; b < this->m_numberOfSamples; b++)
		{
			VectorOfDouble Bb = this->m_dataMatrix.get_column(b);

			double dist = vnl_vector_ssd(Aa, Bb);   // O(m)
			this->m_gramMatrix[a][b] = exp(-dist / (2 * this->m_gamma * this->m_gamma));
			cout << this->m_gramMatrix[a][b] << " "; 
		}
		cout << endl; 
	}
	cout << endl; 

	/** Center Matrix H **/
	MatrixOfDouble identity = MatrixOfDouble(this->m_numberOfSamples, this->m_numberOfSamples, 0.0);
	identity.set_identity();

	/** centralizeKMatrix **/
	MatrixOfDouble Ones = MatrixOfDouble(this->m_numberOfSamples, this->m_numberOfSamples, 1.0);
	MatrixOfDouble H = identity - Ones / this->m_numberOfSamples;

	MatrixOfDouble kCenteredMatrix = H * this->m_gramMatrix * H;

	this->m_eigenVectors.set_size(this->m_numberOfSamples, this->m_numberOfSamples);
	this->m_eigenVals.set_size(this->m_numberOfSamples);
	this->eigenDecomposition(kCenteredMatrix, this->m_eigenVectors, this->m_eigenVals);

	////////////////////////////////////////////////////////////////////

	double eps = 1e-05;  //5/2;
	for (unsigned int i = 0; i < this->m_numberOfSamples; i++)
	{
		if (this->m_eigenVals[i] < eps)
		{
			this->m_eigenVals[i] = eps;
			break;
		}
	}

	// Normalize row such that for each eigenvector a with eigenvalue w we have w(a.a) = 1
	for (unsigned int i = 0; i < this->m_eigenVectors.cols(); i++)
	{		
		this->m_eigenVectors.get_column(i).normalize();
	}
//	this->m_eigenVals /= static_cast<double>(this->m_numberOfSamples - 1);


	/** Compute Kernel Principal Components **/
	this->m_kernelPCs.set_size(this->m_numberOfSamples, this->m_numberOfSamples);
	this->m_kernelPCs = this->m_gramMatrix * this->m_eigenVectors;


	/** selected all eigenValues and eigenVectors **/
	double cum = 0;
	for (unsigned int i = 0; i < this->m_eigenVals.size(); i++)
	{
		cum += this->m_eigenVals.get(i);
		if (cum / this->m_eigenVals.sum() >= 0.97)
		{
			this->m_numberOfRetainedPCs = i + 1; // count number
			break;
		}
	}


	/** extract selected all retainedPCs from eigenvectors and eigenvalues **/
	this->m_eigenVectorsReduced.set_size(this->m_eigenVectors.rows(), this->m_numberOfRetainedPCs);
	this->m_eigenVectorsReduced = this->m_eigenVectors.extract(this->m_eigenVectors.rows(), this->m_numberOfRetainedPCs, 0, 0);

	this->m_eigenValsReduced.set_size(this->m_numberOfRetainedPCs);
	this->m_eigenValsReduced = this->m_eigenVals.extract(this->m_numberOfRetainedPCs, 0);


	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << __FUNCTION__ << " : Reduced Normalized " << std::setprecision(6) << this->m_numberOfRetainedPCs << " eigenValues = : ";
	for (int i = 0; i < this->m_eigenValsReduced.size(); i++)
		std::cout << std::setprecision(6) << this->m_eigenValsReduced[i] << " ";
	std::cout << std::endl;
	std::cout << std::endl;
	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


	/** Reduced Kernel Principal Components **/
	this->m_reducedKernelPCs.set_size(this->m_numberOfSamples, this->m_numberOfRetainedPCs);
	this->m_reducedKernelPCs = this->m_gramMatrix * this->m_eigenVectorsReduced;


	this->m_kPCsMean.set_size(this->m_numberOfRetainedPCs);
	this->m_kPCsStd.set_size(this->m_numberOfRetainedPCs);
	for (int i = 0; i < this->m_reducedKernelPCs.cols(); i++)
	{
		VectorOfDouble tempKPCs = this->m_reducedKernelPCs.get_column(i);
		this->m_kPCsMean[i] = tempKPCs.mean();

		double dist = 0;
		for (int j = 0; j < tempKPCs.size(); j++)
		{
			dist += pow(tempKPCs[j] - this->m_kPCsMean[i], 2);
		}

		this->m_kPCsStd[i] = sqrt(dist / tempKPCs.size());
	}


	/** Reconstruct the input datasets using KPCA **/
	this->constructed_KPCA.set_size(this->m_dataMatrix.rows(), this->m_dataMatrix.cols());
	for (int i = 0; i < this->m_numberOfSamples; i++)
	{
 		VectorOfDouble output = this->backProject(this->m_reducedKernelPCs.get_row(i));   // O (m n ^ 2 )

		this->constructed_KPCA.set_column(i, output);
	}

	if (constructed_KPCA.cols() != m_numberOfSamples || constructed_KPCA.rows() != m_numberOfLandmarks * 3)
	{
		std::cout << "dimensions (number of samples) do not match! " << std::endl;
		return;
	}


	std::cout << __FUNCTION__ << " End ... " << std::endl;
}


void RobustKPCA::RobustKernelLRR()
{
	this->m_modelType = "RKLRR"; 

	// utilize this->m_dataMatrix 
	if (this->m_numberOfSamples == 0)
	{
		std::cout << "Please load the data polys !" << std::endl;
		return;
	}

	cout << __FUNCTION__ << " kernel matrix " << endl;
	this->m_gramMatrix.set_size(this->m_numberOfSamples, this->m_numberOfSamples);
	for (int a = 0; a < this->m_numberOfSamples; a++)
	{
		VectorOfDouble Aa = this->m_dataMatrix.get_column(a);
		for (int b = 0; b < this->m_numberOfSamples; b++)
		{
			VectorOfDouble Bb = this->m_dataMatrix.get_column(b);

			double dist = vnl_vector_ssd(Aa, Bb);
			this->m_gramMatrix[a][b] = exp(-dist / (2 * this->m_gamma * this->m_gamma));
			cout << this->m_gramMatrix[a][b] << " ";
		}
		cout << endl;
	}
	cout << endl;

	/** Center Matrix H **/
	MatrixOfDouble identity = MatrixOfDouble(this->m_numberOfSamples, this->m_numberOfSamples, 0.0);
	identity.set_identity();

	/** centralizeKMatrix **/
	MatrixOfDouble Ones = MatrixOfDouble(this->m_numberOfSamples, this->m_numberOfSamples, 1.0);
	MatrixOfDouble H = identity - Ones / this->m_numberOfSamples;

	MatrixOfDouble kCenteredMatrix = H * this->m_gramMatrix * H;

	this->m_eigenVectors.set_size(this->m_numberOfSamples, this->m_numberOfSamples);
	this->m_eigenVals.set_size(this->m_numberOfSamples);
	this->eigenDecomposition(kCenteredMatrix, this->m_eigenVectors, this->m_eigenVals);

	////////////////////////////////////////////////////////////////////

	double eps = 1e-05;  //5/2;
	for (unsigned int i = 0; i < this->m_numberOfSamples; i++)
	{
		if (this->m_eigenVals[i] < eps)
		{
			this->m_eigenVals[i] = eps;
			break;
		}
	}

	// Normalize row such that for each eigenvector a with eigenvalue w we have w(a.a) = 1
	for (unsigned int i = 0; i < this->m_eigenVectors.cols(); i++)
	{
		this->m_eigenVectors.get_column(i).normalize();
	}


	/** Compute Kernel Principal Components **/
	this->m_kernelPCs.set_size(this->m_numberOfSamples, this->m_numberOfSamples);
	this->m_kernelPCs = this->m_gramMatrix * this->m_eigenVectors;


	/** selected all eigenValues and eigenVectors **/
	double cum = 0;
	for (unsigned int i = 0; i < this->m_eigenVals.size(); i++)
	{
		cum += this->m_eigenVals.get(i);
		if (cum / this->m_eigenVals.sum() >= 0.95)
		{
			this->m_numberOfRetainedPCs = i + 1; // count number
			break;
		}
	}


	/** extract selected all retainedPCs from eigenvectors and eigenvalues **/
	this->m_eigenVectorsReduced.set_size(this->m_eigenVectors.rows(), this->m_numberOfRetainedPCs);
	this->m_eigenVectorsReduced = this->m_eigenVectors.extract(this->m_eigenVectors.rows(), this->m_numberOfRetainedPCs, 0, 0);

	this->m_eigenValsReduced.set_size(this->m_numberOfRetainedPCs);
	this->m_eigenValsReduced = this->m_eigenVals.extract(this->m_numberOfRetainedPCs, 0);


	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << __FUNCTION__ << " : Reduced " << std::setprecision(6) << this->m_numberOfRetainedPCs << " eigenValues = : ";
	for (int i = 0; i < this->m_eigenValsReduced.size(); i++)
		std::cout << std::setprecision(6) << this->m_eigenValsReduced[i] << " ";
	std::cout << std::endl;
	std::cout << std::endl;
	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


	/** Reduced Kernel Principal Components **/
	this->m_reducedKernelPCs.set_size(this->m_numberOfSamples, this->m_numberOfRetainedPCs);
	this->m_reducedKernelPCs = this->m_gramMatrix * this->m_eigenVectorsReduced;


	this->m_kPCsMean.set_size(this->m_numberOfRetainedPCs);
	this->m_kPCsStd.set_size(this->m_numberOfRetainedPCs);
	for (int i = 0; i < this->m_reducedKernelPCs.cols(); i++)
	{
		VectorOfDouble tempKPCs = this->m_reducedKernelPCs.get_column(i);
		this->m_kPCsMean[i] = tempKPCs.mean();

		double dist = 0;
		for (int j = 0; j < tempKPCs.size(); j++)
		{
			dist += pow(tempKPCs[j] - this->m_kPCsMean[i], 2);
		}

		this->m_kPCsStd[i] = sqrt(dist / tempKPCs.size());
	}


	/** Reconstruct the input datasets using KPCA **/
	this->constructed_RobustKernelLRR.set_size(this->m_dataMatrix.rows(), this->m_dataMatrix.cols());
	for (int i = 0; i < this->m_numberOfSamples; i++)
	{
//		VectorOfDouble output = this->backProject(this->m_reducedKernelPCs.get_row(i));

		VectorOfDouble output = this->backProjectAwesome(this->m_reducedKernelPCs.get_row(i), this->m_dataMatrix.get_column(i), 0.7); 

		this->constructed_RobustKernelLRR.set_column(i, output);
	}

	if (constructed_RobustKernelLRR.cols() != m_numberOfSamples || constructed_RobustKernelLRR.rows() != m_numberOfLandmarks * 3)
	{
		std::cout << "dimensions (number of samples) do not match! " << std::endl;
		return;
	}


	std::cout << __FUNCTION__ << " End ... " << std::endl;

}


void RobustKPCA::performNIPS09()
{
	this->m_modelType = "NIPS09"; 

	// utilize this->m_dataMatrix 
	if (this->m_numberOfSamples == 0)
	{
		std::cout << "Please load the data polys !" << std::endl;
		return;
	}

	this->m_gramMatrix.set_size(this->m_numberOfSamples, this->m_numberOfSamples);
	for (int a = 0; a < this->m_numberOfSamples; a++)
	{
		VectorOfDouble Aa = this->m_dataMatrix.get_column(a);
		for (int b = 0; b < this->m_numberOfSamples; b++)
		{
			VectorOfDouble Bb = this->m_dataMatrix.get_column(b);

			double dist = vnl_vector_ssd(Aa, Bb);
			this->m_gramMatrix[a][b] = exp(- dist / (2 * this->m_gamma * this->m_gamma));
		}
	}

	/** Center Matrix H **/
	MatrixOfDouble identity = MatrixOfDouble(this->m_numberOfSamples, this->m_numberOfSamples, 0.0);
	identity.set_identity();

	/** centralizeKMatrix **/
	MatrixOfDouble Ones = MatrixOfDouble(this->m_numberOfSamples, this->m_numberOfSamples, 1.0);
	MatrixOfDouble H = identity - Ones / this->m_numberOfSamples;

	MatrixOfDouble kCenteredMatrix = H * this->m_gramMatrix * H;

	this->m_eigenVectors.set_size(this->m_numberOfSamples, this->m_numberOfSamples);
	this->m_eigenVals.set_size(this->m_numberOfSamples);
	this->eigenDecomposition(kCenteredMatrix, this->m_eigenVectors, this->m_eigenVals);

	////////////////////////////////////////////////////////////////////

	double eps = 1e-05;  //5/2;
	for (unsigned int i = 0; i < this->m_numberOfSamples; i++)
	{
		if (this->m_eigenVals[i] < eps)
		{
			this->m_eigenVals[i] = eps;
			break;
		}
	}

	// Normalize row such that for each eigenvector a with eigenvalue w we have w(a.a) = 1
	for (unsigned int i = 0; i < this->m_eigenVectors.cols(); i++)
	{
		this->m_eigenVectors.get_column(i).normalize(); 
	}

	/** Compute Kernel Principal Components **/
	this->m_kernelPCs.set_size(this->m_numberOfSamples, this->m_numberOfSamples);
	this->m_kernelPCs = this->m_gramMatrix * this->m_eigenVectors;


	/** selected all eigenValues and eigenVectors **/
	double cum = 0;
	for (unsigned int i = 0; i < this->m_eigenVals.size(); i++)
	{
		cum += this->m_eigenVals.get(i);
		if (cum / this->m_eigenVals.sum() >= 0.95)
		{
			this->m_numberOfRetainedPCs = i + 1; // count number
			break;
		}
	}


	/** extract selected all retainedPCs from eigenvectors and eigenvalues **/
	this->m_eigenVectorsReduced.set_size(this->m_eigenVectors.rows(), this->m_numberOfRetainedPCs);
	this->m_eigenVectorsReduced = this->m_eigenVectors.extract(this->m_eigenVectors.rows(), this->m_numberOfRetainedPCs, 0, 0);

	this->m_eigenValsReduced.set_size(this->m_numberOfRetainedPCs);
	this->m_eigenValsReduced = this->m_eigenVals.extract(this->m_numberOfRetainedPCs, 0);


	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << __FUNCTION__ << " : Reduced " << this->m_numberOfRetainedPCs << " eigenValues = : ";
	for (int i = 0; i < this->m_eigenValsReduced.size(); i++)
		std::cout << std::setprecision(6) << this->m_eigenValsReduced[i] << " ";
	std::cout << std::endl;
	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


	/** Reduced Kernel Principal Components **/
	this->m_reducedKernelPCs.set_size(this->m_numberOfSamples, this->m_numberOfRetainedPCs);
	this->m_reducedKernelPCs = this->m_gramMatrix * this->m_eigenVectorsReduced;

	this->m_kPCsMean.set_size(this->m_numberOfRetainedPCs);
	this->m_kPCsStd.set_size(this->m_numberOfRetainedPCs);
	for (int i = 0; i < this->m_reducedKernelPCs.cols(); i++)
	{
		VectorOfDouble tempKPCs = this->m_reducedKernelPCs.get_column(i);
		this->m_kPCsMean[i] = tempKPCs.mean();

		double dist = 0;
		for (int j = 0; j < tempKPCs.size(); j++)
		{
			dist += pow(tempKPCs[j] - this->m_kPCsMean[i], 2);
		}

		this->m_kPCsStd[i] = sqrt(dist / tempKPCs.size());
	}


	/** Reconstruct the input datasets using KPCA **/
	this->constructed_NIPS09.set_size(this->m_dataMatrix.rows(), this->m_dataMatrix.cols());
	for (int i = 0; i < this->m_numberOfSamples; i++)
	{
//		VectorOfDouble output = this->backProjectAwesome(this->m_reducedKernelPCs.get_row(i)); 

		VectorOfDouble output = this->backProjectNIPS09(this->m_dataMatrix.get_column(i), 0.09); 

		this->constructed_NIPS09.set_column(i, output);
	}

	if (constructed_NIPS09.cols() != m_numberOfSamples || constructed_NIPS09.rows() != m_numberOfLandmarks * 3)
	{
		std::cout << "dimensions (number of samples) do not match! " << std::endl;
		return;
	}

	std::cout << __FUNCTION__ << " End ... " << std::endl;
}


void RobustKPCA::RKPCA(int _iterNum, double _proportion)
{
	cout << __FUNCTION__ << endl; 

	this->m_modelType = "RKPCA";

	if (this->m_numberOfSamples == 0)
	{
		std::cout << " Please load the data polys !" << std::endl;
		return;
	}

	unsigned int m = this->m_dataMatrix.rows();
	unsigned int n = this->m_numberOfSamples;

	this->m_gramMatrix.set_size(n, n);
	for (int a = 0; a < this->m_numberOfSamples; a++)  // O(m n^2 ) 
	{
		VectorOfDouble Aa = this->m_dataMatrix.get_column(a);
		for (int b = 0; b < this->m_numberOfSamples; b++)
		{
			VectorOfDouble Bb = this->m_dataMatrix.get_column(b);

			double dist = vnl_vector_ssd(Aa, Bb);
			this->m_gramMatrix[a][b] = exp(-dist / (2 * this->m_gamma * this->m_gamma));
		}
	}

	//********************************************************************//

	MatrixOfDouble m_K = this->m_gramMatrix;

	MatrixOfDouble m_L(m, n, 0.0);
	MatrixOfDouble m_S(m, n, 0.0);
	MatrixOfDouble Y_M = m_dataMatrix; // for sparsity 

	vnl_svd< double > svdY_M(Y_M);
	double lambda_M = 1.0 / sqrt(m);
	double norm2_M = svdY_M.sigma_max();
	double normInf_M = Y_M.operator_inf_norm() / lambda_M;

	Y_M = Y_M / std::fmax(norm2_M, normInf_M);

	/**********************************************************************/

	double mu_M = 1.25 / norm2_M;  
	double mu_max_M = mu_M * 1e7;
	double rho_M = 1.6; 

	int iter = 0;

	itk::TimeProbe itkClock; 
	itkClock.Start(); 

	while (iter < _iterNum)
	{
		MatrixOfDouble pre_m_L = m_L; 

		/************************** Finding m_S *******************************************/

		MatrixOfDouble temp_M = m_dataMatrix - m_L + (1.0 / mu_M) * Y_M;

		for (int eCol = 0; eCol < temp_M.cols(); eCol++)   // O ( mn ) 
		{
			for (int eRow = 0; eRow < temp_M.rows(); eRow++)
			{
				m_S[eRow][eCol] = std::fmax(temp_M[eRow][eCol] - lambda_M / mu_M, 0.0)
					+ std::fmin(temp_M[eRow][eCol] + lambda_M / mu_M, 0.0);
			}
		}

		/*************************** Finding m_K ******************************************/

		MatrixOfDouble temp_L = m_dataMatrix - m_S + (1.0 / mu_M) * Y_M;   // m_L = m_dataMatrix - m_S;

		MatrixOfDouble pre_K(n, n, 0.0); 
		for (int a = 0; a < n; a++)
		{
			VectorOfDouble Aa = temp_L.get_column(a);
			for (int b = 0; b < n; b++)
			{
				VectorOfDouble Bb = temp_L.get_column(b);

				double dist = vnl_vector_ssd(Aa, Bb);    // O ( m n ^ 2 ) 
				pre_K[a][b] = exp(-dist / (2 * this->m_gamma * this->m_gamma));
			}
		}

		/*************************************************/

		MatrixOfDouble A = this->LowRankModelingKernelMatrix(temp_L, pre_K, m_K);   // O(n^3 + mn)
		MatrixOfDouble C = m_dataMatrix - m_S + (1.0 / mu_M) * Y_M;

		for (int i = 0; i < m_dataMatrix.cols(); i++)
		{
			double kzx = m_K.get_column(i).sum(); 

			if (kzx == 0)
			{
				cout << " WARNING kzx = 0 ... " << endl; 

				continue; 
			}

			// mu_M -> zero, the influence of C decreases 
			VectorOfDouble currVector = (A.get_column(i) + mu_M * C.get_column(i)) / (kzx + mu_M);   

			m_L.set_column(i, currVector);
		}
	
		/*****************************************************/

		/**********************************************************************************/

		MatrixOfDouble Z_M = m_dataMatrix - m_L - m_S;

		Y_M = Y_M + mu_M * Z_M;
		mu_M = std::fmin(mu_M * rho_M, mu_max_M);

		/**********************************************************************************/
		
		double stopCriterion = Z_M.frobenius_norm() / m_dataMatrix.frobenius_norm();
		double stopTermine = (pre_m_L - m_L).frobenius_norm() / pre_m_L.frobenius_norm();

		int e = 0;
		int z = 0; 
		for (int eCol = 0; eCol < m_S.cols(); eCol++)
		{
			for (int eRow = 0; eRow < m_S.rows(); eRow++)
			{
				if (abs(m_S[eRow][eCol]) > 0)
					e++;

				if (abs(Z_M[eRow][eCol] > 0))
					z++; 
			}
		}

		if (stopCriterion < 1e-3 / 2)
		{
			std::cout << "=================================================================" << endl; 
			std::cout << " Converged true => ";
			std::cout << " iter : " << iter << ", sparse entries = " << e << ", z = " << z << std::setprecision(5) << " StopCriterion =  "
				<< stopCriterion << " , stopTermine = " << stopTermine << std::endl;
			break; 
		}


		if (stopTermine < 1e-3 / 2)
		{
			std::cout << "=================================================================" << endl;
			std::cout << " Low-rank converged => ";
			std::cout << " iter : " << iter << ", sparse entries = " << e << ", z = " << z << std::endl;

			break; 
		}

		std::cout << endl; 
		std::cout << " iter : " << iter << ", sparse entries = " << e << ", z = " << z << std::setprecision(5) << " StopCriterion =  " 
			<< stopCriterion << " , stopTermine = " << stopTermine << std::endl;
		std::cout << endl; 

		iter++;

	}

	itkClock.Stop();
	std::cout << __FUNCTION__ << " : Time for RKPCA for " << this->m_numberOfLandmarks << " landmarks, " << this->m_numberOfSamples << " datasets are " << itkClock.GetMean() << " seconds ." << std::endl;
	std::cout << std::endl;

	//***************************************************************************//

	cout << __FUNCTION__ << " update and print the new kernel matrix ... " << endl;
	for (int a = 0; a < n; a++)
	{
		for (int b = 0; b < n; b++)
		{
			this->m_gramMatrix[a][b] = m_K[a][b];
			cout << this->m_gramMatrix[a][b] << " ";
		}
		cout << endl;
	}

	/** Center Matrix H **/
	MatrixOfDouble identity = MatrixOfDouble(n, n, 0.0);
	identity.set_identity();

	/** centralizeKMatrix **/
	MatrixOfDouble Ones = MatrixOfDouble(n, n, 1.0);
	MatrixOfDouble H = identity - Ones / n;
	MatrixOfDouble kCenteredMatrix = H * this->m_gramMatrix * H;

	this->m_eigenVectors.set_size(n, n);
	this->m_eigenVals.set_size(n);
	this->eigenDecomposition(kCenteredMatrix, this->m_eigenVectors, this->m_eigenVals); 

	double eps = 1e-05;  //5/2;
	for (unsigned int i = 0; i < n; i++)
	{
		if (this->m_eigenVals[i] < eps)
		{
			this->m_eigenVals[i] = eps;
			break;
		}
	}

	for (unsigned int i = 0; i < this->m_eigenVectors.cols(); i++)
	{
		this->m_eigenVectors.get_column(i).normalize(); 
	}


	/** Compute Kernel Principal Components **/
	this->m_kernelPCs.set_size(n, n);
	this->m_kernelPCs = this->m_gramMatrix * this->m_eigenVectors;


	/** selected all eigenValues and eigenVectors **/
	double cum = 0;
	for (unsigned int i = 0; i < this->m_eigenVals.size(); i++)
	{
		cum += this->m_eigenVals.get(i);

		if (cum / this->m_eigenVals.sum() >= 0.95)
		{
			this->m_numberOfRetainedPCs = i + 1; // count number
			break;
		}
	}

	/** extract selected all retainedPCs from eigenvectors and eigenvalues **/
	this->m_eigenVectorsReduced.set_size(this->m_eigenVectors.rows(), this->m_numberOfRetainedPCs);
	this->m_eigenVectorsReduced = this->m_eigenVectors.extract(this->m_eigenVectors.rows(), this->m_numberOfRetainedPCs, 0, 0);

	this->m_eigenValsReduced.set_size(this->m_numberOfRetainedPCs);
	this->m_eigenValsReduced = this->m_eigenVals.extract(this->m_numberOfRetainedPCs, 0);


	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	cout << endl;
	cout << __FUNCTION__ << " : " << std::setprecision(6) << this->m_eigenValsReduced.size() << " eigenValues = " << endl;
	for (int i = 0; i < this->m_eigenValsReduced.size(); i++)
		cout << this->m_eigenValsReduced[i] << " ";
	cout << endl;
	cout << endl;
	//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


	/** Reduced Kernel Principal Components **/
	this->m_reducedKernelPCs.set_size(n, this->m_numberOfRetainedPCs);
	this->m_reducedKernelPCs = this->m_gramMatrix * this->m_eigenVectorsReduced;  // row vectors stacked [ n * k ] 

	this->m_kPCsMean.set_size(this->m_numberOfRetainedPCs);
	this->m_kPCsStd.set_size(this->m_numberOfRetainedPCs);
	for (int i = 0; i < this->m_reducedKernelPCs.cols(); i++)
	{
		VectorOfDouble tempKPCs = this->m_reducedKernelPCs.get_column(i);
		this->m_kPCsMean[i] = tempKPCs.mean();

		double dist = 0;
		for (int j = 0; j < tempKPCs.size(); j++)
		{
			dist += pow(tempKPCs[j] - this->m_kPCsMean[i], 2);
		}

		this->m_kPCsStd[i] = sqrt(dist / tempKPCs.size());
	}

	//////////// Update m_dataMatrix with constructed_RobustKernelPCA //////////////////////////////////////////////////////
	
	if (_proportion > 0)
	{
		std::cout << __FUNCTION__ << " : Upgrade the training dataMatrix for back projection " << std::endl;
		for (int i = 0; i < this->m_numberOfSamples; i++)
		{
			this->m_dataMatrix.set_column(i, m_L.get_column(i));
		}
	}

	this->constructed_RKPCA = m_L; 

	if (constructed_RKPCA.cols() != n || constructed_RKPCA.rows() != m_numberOfLandmarks * 3)
	{
		std::cout << " dimensions (number of samples) do not match! " << std::endl;
		return;
	}

	std::cout << __FUNCTION__ << " : Finish RKPCA [RKPCA] -> Got constructed_RKPCA " << std::endl;


	//////////////////////////////// Update m_meanShapeVector & m_meanShape ///////////////////////////////////////////////

	this->m_meanShapeVector.fill(0.0);

	for (int i = 0; i < this->m_numberOfSamples; i++)
		this->m_meanShapeVector += this->m_dataMatrix.get_column(i);

	this->m_meanShapeVector /= this->m_numberOfSamples;

	cout << __FUNCTION__ << " End ... " << endl;
}


void RobustKPCA::Miccai17()
{
	/* robust kernel matrix construction + normal back projection */
	this->m_modelType = "Miccai17";

	/** Perform Kernel PCA : Get m_gramMatrix **/
	if (this->m_numberOfSamples == 0)
	{
		std::cout << " Please load the data polys !" << std::endl;
		return;
	}

	unsigned int n = this->m_numberOfSamples;

	this->m_gramMatrix.set_size(n, n);
	for (int a = 0; a < this->m_numberOfSamples; a++)
	{
		VectorOfDouble Aa = this->m_dataMatrix.get_column(a);
		for (int b = 0; b < this->m_numberOfSamples; b++)
		{
			VectorOfDouble Bb = this->m_dataMatrix.get_column(b);

			double dist = vnl_vector_ssd(Aa, Bb);
			this->m_gramMatrix[a][b] = exp(-dist / (2 * this->m_gamma * this->m_gamma));
		}
	}

	// print MICCAI-17 original kernel matrix 
	cout << __FUNCTION__ << ": Original Kernel Matrix ... " << endl; 
	for (int i = 0; i < this->m_gramMatrix.rows(); i++)
	{
		for (int j = 0; j < this->m_gramMatrix.cols(); j++)
			cout << this->m_gramMatrix[i][j] << " "; 
		cout << endl; 
	}
	cout << endl; 
	cout << "-------------------------------------------------------------------" << endl; 
	cout << endl; 



	//////////// Perform Robust KPCA //////////////////////////////////////////////////////////////////////

	double lambda = 1.0 / sqrt(n);  // lambda only influence the sparse matrix computation, tuned
	double tol1 = 1e-10;
	double tol2 = 1e-5;
	int maxIter = 500;

	vnl_svd< double > svdK(this->m_gramMatrix);
	double norm2 = svdK.sigma_max();
	double normInf = this->m_gramMatrix.operator_inf_norm() / lambda;
	double normDual = std::fmax(norm2, normInf);
	double dNorm = this->m_gramMatrix.frobenius_norm();


	MatrixXd kernelM = MatrixXd::Zero(n, n);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			kernelM(i, j) = this->m_gramMatrix[i][j];
		}
	}

	/* Initialize L , Y = 0, P = I */
	MatrixXd m_L = MatrixXd::Zero(n, n);
	MatrixXd m_Y = MatrixXd::Zero(n, n);
	MatrixXd m_E = MatrixXd::Zero(n, n);
	MatrixXd m_I = MatrixXd::Identity(n, n);
	MatrixXd sqrtXTX = MatrixXd::Identity(n, n);
	MatrixXd m_L_old = MatrixXd::Zero(n, n);

	m_Y = kernelM / normDual;

	double mu = 1.25 / norm2; 
	double mu_max = mu * 1e6;
	double rho = 1.6;

	int iter = 0;
	double converged = false;

	itk::TimeProbe itkClock; 
	itkClock.Start(); 

	//==========================================================================================
	/* Optimization via ALM */
	while (!converged)
	{
		iter++;

		////////////////////////////////////////////////////////////////////////
		/* m_E via SVT */

		MatrixXd deltaE = kernelM - m_L + (1.0 / mu) * m_Y;

		for (int eCol = 0; eCol < n; eCol++)
		{
			for (int eRow = 0; eRow < n; eRow++)
			{
				m_E(eRow, eCol) = std::fmax(deltaE(eRow, eCol) - lambda / mu, 0.0)
					+ std::fmin(deltaE(eRow, eCol) + lambda / mu, 0.0);
			}
		}

		////////////////////////////////////////////////////////////////////////////
		/* Get m_L (Z) */

		MatrixXd tempA = mu * m_I;
		arma::mat SylvesterA = zeros<mat>(n, n);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
				SylvesterA(i, j) = tempA(i, j);
		}

		MatrixXd temp209 = kernelM - m_E + mu * mu * m_I;  // avoid to be all zeros 
		MatrixXd TempXTX = temp209.transpose() * temp209;

		Eigen::EigenSolver< MatrixXd > eig(TempXTX);
		MatrixXd eigVectors = eig.eigenvectors().real();
		VectorXd eigValues = eig.eigenvalues().real();
		MatrixXd eigVectorsInverse = eig.eigenvectors().real().inverse();

		MatrixXd sqrtEigValues = MatrixXd::Zero(n, n);
		for (int i = 0; i < n; i++)
		{
			if (eigValues(i) >= 0)
				sqrtEigValues(i, i) = sqrt(eigValues(i));
			else
				sqrtEigValues(i, i) = 0; 
		}

		sqrtXTX = eigVectors * sqrtEigValues * eigVectorsInverse;

		//////////////////////////////////////////////////

		MatrixXd tempB = sqrtXTX.inverse();
		arma::mat SylvesterB = zeros<mat>(n, n);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
				SylvesterB(i, j) = tempB(i, j);
		}

		MatrixXd deltaL = kernelM - m_E + (1.0 / mu) * m_Y;
		MatrixXd tempC = (-1) * mu * deltaL;
		arma::mat SylvesterC = zeros<mat>(n, n);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
				SylvesterC(i, j) = tempC(i, j);
		}

		arma::mat X = zeros<mat>(n, n);
		X = arma::syl(SylvesterA, SylvesterB, SylvesterC);

		// convert to L 
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
				m_L(i, j) = X(i, j);
		}

		//////////////////////////////////////////////////////////////////////////////

	    // shrinkage kernel_m_L 
		Eigen::JacobiSVD< MatrixXd > svd(m_L, ComputeThinU | ComputeThinV);

		VectorXd singularValues = svd.singularValues();

		int svp = 0;
		for (int i = 0; i < n; i++)
		{
			if (singularValues(i) > 1.0 / mu)
				svp++;
		}

		MatrixXd Sigmas = MatrixXd::Zero(svp, svp);
		for (int i = 0; i < svp; i++)
		{
			Sigmas(i, i) = singularValues(i) - 1.0 / mu;
		}

		MatrixXd Left = MatrixXd::Zero(n, svp);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < svp; j++)
				Left(i, j) = svd.matrixU()(i, j);
		}
		MatrixXd Right = MatrixXd::Zero(n, svp);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < svp; j++)
				Right(i, j) = svd.matrixV()(i, j);
		}

		m_L = Left * Sigmas * Right.transpose();

		//////////////////////////////////////////////////////////////////////////////
		/* Update and Convergence Analysis */

		MatrixXd deltaK = kernelM - m_L - m_E;

		int test = 0;
		for (int i = 0; i < deltaK.cols(); i++)
		{
			for (int j = 0; j < deltaK.rows(); j++)
			{
				if (abs(deltaK(j, i)) > 0)
					test++;
			}
		}

//		std::cout << " iter = " << iter << std::setprecision(3) << ", deltaK = " << test << " , 1/mu = " << 1.0 / mu << " , svp = " << svp << ", sparse = " << test << endl;

		m_Y = m_Y + mu * deltaK;
		mu = std::fmin(mu * rho, mu_max);

		MatrixOfDouble Z(n, n, 0.0);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
				Z[i][j] = deltaK(i, j);
		}

		double stopCriterion1 = Z.frobenius_norm() / dNorm;

		if (!converged && iter >= maxIter)
		{
			std::cout << "Maximum iterations reached" << std::endl;
			converged = true;
		}


		MatrixOfDouble DL(n, n, 0);
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
				DL[i][j] = m_L_old(i, j) - m_L(i, j);
		}


		double stopCriterion2 = DL.frobenius_norm();

		if (stopCriterion1 < tol1 && stopCriterion2 < tol2)
		{
			converged = true;
			std::cout << "Both stopCriterions have been reached : " << stopCriterion2 << std::endl;
			break;
		}

		m_L_old = m_L;

	}

	std::cout << __FUNCTION__ << " : time for kernel training is " << std::setprecision(4) << itkClock.GetMean() << " seconds .. " << endl; 

	//==========================================================================================
	/* Return the low-rank subspace represented by L_{k+1} */

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			this->m_gramMatrix[i][j] = m_L(i, j);
		}
	}

	/** Center Matrix H **/
	MatrixOfDouble identity = MatrixOfDouble(n, n, 0.0);
	identity.set_identity();

	/** centralizeKMatrix **/
	MatrixOfDouble Ones = MatrixOfDouble(n, n, 1.0);
	MatrixOfDouble H = identity - Ones / n;
	MatrixOfDouble kCenteredMatrix = H * this->m_gramMatrix * H;

	this->m_eigenVectors.set_size(n, n); 
	this->m_eigenVals.set_size(n); 
	this->eigenDecomposition(kCenteredMatrix, this->m_eigenVectors, this->m_eigenVals); 

	double eps = 1e-05;  //5/2;
	for (unsigned int i = 0; i < n; i++)
	{
		if (this->m_eigenVals[i] < eps)
		{
			this->m_eigenVals[i] = eps;
			break;
		}
	}

	// Normalize row such that for each eigenvector a with eigenvalue w we have w(a.a) = 1
	for (unsigned int i = 0; i < this->m_eigenVectors.cols(); i++)
	{
		this->m_eigenVectors.get_column(i).normalize(); 
	}


	/** Compute Kernel Principal Components **/
	this->m_kernelPCs.set_size(n, n);
	this->m_kernelPCs = this->m_gramMatrix * this->m_eigenVectors;


	/** selected all eigenValues and eigenVectors **/
	double cum = 0;
	for (unsigned int i = 0; i < this->m_eigenVals.size(); i++)
	{
		cum += this->m_eigenVals.get(i);

		if (cum / this->m_eigenVals.sum() >= 0.97)
		{
			this->m_numberOfRetainedPCs = i + 1; // count number
			break;
		}
	}

	/** extract selected all retainedPCs from eigenvectors and eigenvalues **/
	this->m_eigenVectorsReduced.set_size(this->m_eigenVectors.rows(), this->m_numberOfRetainedPCs);
	this->m_eigenVectorsReduced = this->m_eigenVectors.extract(this->m_eigenVectors.rows(), this->m_numberOfRetainedPCs, 0, 0);

	this->m_eigenValsReduced.set_size(this->m_numberOfRetainedPCs);
	this->m_eigenValsReduced = this->m_eigenVals.extract(this->m_numberOfRetainedPCs, 0);


	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	cout << endl;
	cout << __FUNCTION__ << " : MICCAI17 " << std::setprecision(6) << this->m_eigenValsReduced.size() << " eigenValues = " << endl;
	for (int i = 0; i < this->m_eigenValsReduced.size(); i++)
		cout << this->m_eigenValsReduced[i] << " ";
	cout << endl;
	cout << endl;
	//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


	/** Reduced Kernel Principal Components **/
	this->m_reducedKernelPCs.set_size(n, this->m_numberOfRetainedPCs);
	this->m_reducedKernelPCs = this->m_gramMatrix * this->m_eigenVectorsReduced;  // row vectors stacked [ n * k ] 

	this->m_kPCsMean.set_size(this->m_numberOfRetainedPCs);
	this->m_kPCsStd.set_size(this->m_numberOfRetainedPCs);
	for (int i = 0; i < this->m_reducedKernelPCs.cols(); i++)
	{
		VectorOfDouble tempKPCs = this->m_reducedKernelPCs.get_column(i);
		this->m_kPCsMean[i] = tempKPCs.mean();

		double dist = 0;
		for (int j = 0; j < tempKPCs.size(); j++)
		{
			dist += pow(tempKPCs[j] - this->m_kPCsMean[i], 2);
		}

		this->m_kPCsStd[i] = sqrt(dist / tempKPCs.size());
	}


	this->constructed_Miccai17.set_size(this->m_dataMatrix.rows(), this->m_dataMatrix.cols());
	for (int i = 0; i < n; i++)
	{
		VectorOfDouble output = this->backProjectRobust(this->m_reducedKernelPCs.get_row(i), this->m_dataMatrix.get_column(i));
//		VectorOfDouble output = this->backProject(this->m_reducedKernelPCs.get_row(i)); 
		this->constructed_Miccai17.set_column(i, output);
	}

	if (constructed_RobustKernelPCA.cols() != n || constructed_RobustKernelPCA.rows() != m_numberOfLandmarks * 3)
	{
		std::cout << " dimensions (number of samples) do not match! " << std::endl;
		return;
	}

	std::cout << __FUNCTION__ << " : Finish performRobustKPCA in [RobustKPCA] -> Got constructed_RobustKPCA " << std::endl;


	//////////// Update m_dataMatrix with constructed_RobustKernelPCA //////////////////////////////////////////////////////
	for (int i = 0; i < this->m_numberOfSamples; i++)
	{
		this->m_dataMatrix.set_column(i, this->constructed_RobustKernelPCA.get_column(i));
	}

	std::cout << __FUNCTION__ << " : Upgrade the training dataMatrix for back projection " << std::endl;

	//////////////////////////////// Update m_meanShapeVector & m_meanShape ///////////////////////////////////////////////

	this->m_meanShapeVector.fill(0.0);

	for (int i = 0; i < this->m_numberOfSamples; i++)
		this->m_meanShapeVector += this->m_dataMatrix.get_column(i);

	this->m_meanShapeVector /= this->m_numberOfSamples;

	cout << __FUNCTION__ << " End ... " << endl;

}


//void RobustKPCA::RobustKernelPCA()
//{
//	this->m_modelType = "RKPCA"; 
//
//	/** Perform Kernel PCA : Get m_gramMatrix **/
//	if (this->m_numberOfSamples == 0)
//	{
//		std::cout << " Please load the data polys !" << std::endl;
//		return;
//	}
//
//	unsigned int n = this->m_numberOfSamples;
//
//	this->m_gramMatrix.set_size(n, n);
//	for (int a = 0; a < this->m_numberOfSamples; a++)
//	{
//		VectorOfDouble Aa = this->m_dataMatrix.get_column(a);
//		for (int b = 0; b < this->m_numberOfSamples; b++)
//		{
//			VectorOfDouble Bb = this->m_dataMatrix.get_column(b);
//
//			double dist = vnl_vector_ssd(Aa, Bb);
//			this->m_gramMatrix[a][b] = exp(- dist / (2 * this->m_gamma * this->m_gamma));
//		}
//	}
//
//
//	//////////// Perform Robust KPCA //////////////////////////////////////////////////////////////////////
//
//	double lambda = 1.0 / sqrt(n);  // 1.0 / sqrt((double)std::fmax(this->m_numberOfLandmarks, n));  // lambda only influence the sparse matrix computation, tuned
//	double tol1 = 1e-10;
//	double tol2 = 1e-5;
//	int maxIter = 500;
//
//	vnl_svd< double > svdK(this->m_gramMatrix);
//	double norm2 = svdK.sigma_max();
//	double normInf = this->m_gramMatrix.operator_inf_norm() / lambda;
//	double normDual = std::fmax(norm2, normInf);
//	double dNorm = this->m_gramMatrix.frobenius_norm();
//
//
//	MatrixXd kernelM = MatrixXd::Zero(n, n);
//	for (int i = 0; i < n; i++)
//	{
//		for (int j = 0; j < n; j++)
//		{
//			kernelM(i, j) = this->m_gramMatrix[i][j];
//		}
//	}
//
//	/* Initialize L , Y = 0, P = I */
//	MatrixXd m_L = MatrixXd::Zero(n, n);
//	MatrixXd m_Y = MatrixXd::Zero(n, n);
//	MatrixXd m_E = MatrixXd::Zero(n, n);
//	MatrixXd m_I = MatrixXd::Identity(n, n);
//	MatrixXd sqrtXTX = MatrixXd::Identity(n, n);
//	MatrixXd m_L_old = MatrixXd::Zero(n, n);
//
//	m_Y = kernelM / normDual;
//
//	double mu = 1.25 / norm2; 
//	double mu_max = mu * 1e6;
//	double rho = 1.6;
//	int iter = 0;
//	double converged = false;
//
//	//==========================================================================================
//	/* Optimization via ALM */
//	while (!converged)
//	{
//		iter++;
//
//		////////////////////////////////////////////////////////////////////////
//		/* m_E via SVT */
//
//		MatrixXd deltaE = kernelM - m_L + (1.0 / mu) * m_Y;
//
//		for (int eCol = 0; eCol < n; eCol++)
//		{
//			for (int eRow = 0; eRow < n; eRow++)
//			{
//				m_E(eRow, eCol) = std::fmax(deltaE(eRow, eCol) - 1 / mu, 0.0)
//					+ std::fmin(deltaE(eRow, eCol) + 1 / mu, 0.0);
//			}
//		}
//
//
//		////////////////////////////////////////////////////////////////////////////
//		/* Get m_L (Z) */
//
//		MatrixXd tempA = mu * m_I;
//		arma::mat SylvesterA = zeros<mat>(n, n);
//		for (int i = 0; i < n; i++)
//		{
//			for (int j = 0; j < n; j++)
//				SylvesterA(i, j) = tempA(i, j);
//		}
//
//		MatrixXd temp209 = kernelM - m_E + mu * mu * m_I;  // avoid to be all zeros 
//		MatrixXd TempXTX = temp209.transpose() * temp209;
//
//		Eigen::EigenSolver< MatrixXd > eig(TempXTX);
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
//
//		//////////////////////////////////////////////////
//
//		MatrixXd tempB = sqrtXTX.inverse();
//		arma::mat SylvesterB = zeros<mat>(n, n);
//		for (int i = 0; i < n; i++)
//		{
//			for (int j = 0; j < n; j++)
//				SylvesterB(i, j) = tempB(i, j);
//		}
//
//		MatrixXd deltaL = kernelM - m_E + (1.0 / mu) * m_Y;
//		MatrixXd tempC = (-1) * mu * deltaL;
//		arma::mat SylvesterC = zeros<mat>(n, n);
//		for (int i = 0; i < n; i++)
//		{
//			for (int j = 0; j < n; j++)
//				SylvesterC(i, j) = tempC(i, j);
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
//		//////////////////////////////////////////////////////////////////////////////
//
//	/*	// m_L = kernelM - m_E + (1.0 / mu) * m_Y; 
//	    Eigen::JacobiSVD< MatrixXd > svd(m_L, ComputeThinU | ComputeThinV);
//
//		VectorXd singularValues = svd.singularValues();
//
//		int svp = 0;
//		for (int i = 0; i < n; i++)
//		{
//			if (singularValues(i) > 1.0 / mu)
//				svp++;
//		}
//
//		MatrixXd Sigmas = MatrixXd::Zero(svp, svp);
//		for (int i = 0; i < svp; i++)
//		{
//			Sigmas(i, i) = singularValues(i) - 1.0 / mu;
//		}
//
//		MatrixXd Left = MatrixXd::Zero(n, svp);
//		for (int i = 0; i < n; i++)
//		{
//			for (int j = 0; j < svp; j++)
//				Left(i, j) = svd.matrixU()(i, j);
//
//		}
//		MatrixXd Right = MatrixXd::Zero(n, svp);
//		for (int i = 0; i < n; i++)
//		{
//			for (int j = 0; j < svp; j++)
//				Right(i, j) = svd.matrixV()(i, j);
//		}
//
//		m_L = Left * Sigmas * Right.transpose();*/
//
//		//////////////////////////////////////////////////////////////////////////////
//		/* Update and Convergence Analysis */
//
//		MatrixXd deltaK = kernelM - m_L - m_E;
//
//		int test = 0;
//		for (int i = 0; i < deltaK.cols(); i++)
//		{
//			for (int j = 0; j < deltaK.rows(); j++)
//			{
//				if (abs(deltaK(j, i)) <= 0)
//					test++;
//			}
//		}
//		std::cout << "iter = " << iter << ", deltaK = " << test << " , mu = " << mu << " , ";
//
//		m_Y = m_Y + mu * deltaK;
//		mu = std::fmin(mu * rho, mu_max);
//
//		MatrixOfDouble Z(n, n, 0.0);
//		for (int i = 0; i < n; i++)
//		{
//			for (int j = 0; j < n; j++)
//				Z[i][j] = deltaK(i, j);
//		}
//
//		double stopCriterion1 = Z.frobenius_norm() / dNorm;
//
//		if (!converged && iter >= maxIter)
//		{
//			std::cout << "Maximum iterations reached" << std::endl;
//			converged = true;
//		}
//
//
//		MatrixOfDouble DL(n, n, 0);
//		for (int i = 0; i < n; i++)
//		{
//			for (int j = 0; j < n; j++)
//				DL[i][j] = m_L_old(i, j) - m_L(i, j);
//		}
//
//
//		double stopCriterion2 = DL.frobenius_norm();
//
//		if (stopCriterion1 < tol1 && stopCriterion2 < tol2)
//		{
//			converged = true;
//			std::cout << "Both stopCriterions have been reached : " << stopCriterion2 << std::endl;
//			break;
//		}
//
//		m_L_old = m_L;
//
//		std::cout << " stop1 = " << stopCriterion1 << " , stop2 = " << stopCriterion2 << std::endl;
//
//	}
//
//	//==========================================================================================
//	/* Return the low-rank subspace represented by L_{k+1} */
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
//	/** Center Matrix H **/
//	MatrixOfDouble identity = MatrixOfDouble(n, n, 0.0);
//	identity.set_identity();
//
//	/** centralizeKMatrix **/
//	MatrixOfDouble Ones = MatrixOfDouble(n, n, 1.0);
//	MatrixOfDouble H = identity - Ones / n;
//	MatrixOfDouble kCenteredMatrix = H * this->m_gramMatrix * H;
//
//	this->m_eigenVectors.set_size(n, n); 
//	this->m_eigenVals.set_size(n); 
//	this->eigenDecomposition(kCenteredMatrix, this->m_eigenVectors, this->m_eigenVals); 
//
//	double eps = 1e-05;  //5/2;
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
//		this->m_eigenVectors.get_column(i).normalize(); 
//	}
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
//
//		if (cum / this->m_eigenVals.sum() >= 0.97)
//		{
//			this->m_numberOfRetainedPCs = i + 1; // count number
//			break;
//		}
//	}
//
//	/** extract selected all retainedPCs from eigenvectors and eigenvalues **/
//	this->m_eigenVectorsReduced.set_size(this->m_eigenVectors.rows(), this->m_numberOfRetainedPCs);
//	this->m_eigenVectorsReduced = this->m_eigenVectors.extract(this->m_eigenVectors.rows(), this->m_numberOfRetainedPCs, 0, 0);
//
//	this->m_eigenValsReduced.set_size(this->m_numberOfRetainedPCs);
//	this->m_eigenValsReduced = this->m_eigenVals.extract(this->m_numberOfRetainedPCs, 0);
//
//
//	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//	cout << endl;
//	cout << __FUNCTION__ << " : reduced " << std::setprecision(6) << this->m_eigenValsReduced.size() << " eigenValues = " << endl;
//	for (int i = 0; i < this->m_eigenValsReduced.size(); i++)
//		cout << this->m_eigenValsReduced[i] << " ";
//	cout << endl;
//	cout << endl;
//	//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//
//
//	/** Reduced Kernel Principal Components **/
//	this->m_reducedKernelPCs.set_size(n, this->m_numberOfRetainedPCs);
//	this->m_reducedKernelPCs = this->m_gramMatrix * this->m_eigenVectorsReduced;  // row vectors stacked [ n * k ] 
//
//	this->m_kPCsMean.set_size(this->m_numberOfRetainedPCs);
//	this->m_kPCsStd.set_size(this->m_numberOfRetainedPCs);
//	for (int i = 0; i < this->m_reducedKernelPCs.cols(); i++)
//	{
//		VectorOfDouble tempKPCs = this->m_reducedKernelPCs.get_column(i); 
//		this->m_kPCsMean[i] = tempKPCs.mean(); 
//
//		double dist = 0; 
//		for (int j = 0; j < tempKPCs.size(); j++)
//		{
//			dist += pow(tempKPCs[j] - this->m_kPCsMean[i], 2); 
//		}
//
//		this->m_kPCsStd[i] = sqrt(dist / tempKPCs.size());
//	}
//
//
//	this->constructed_RobustKernelPCA.set_size(this->m_dataMatrix.rows(), this->m_dataMatrix.cols());
//	for (int i = 0; i < n; i++)
//	{
//		VectorOfDouble output = this->backProjectRobust(this->m_reducedKernelPCs.get_row(i), this->m_dataMatrix.get_column(i));
//
////		VectorOfDouble output = this->backProject(this->m_reducedKernelPCs.get_row(i)); 
//		this->constructed_RobustKernelPCA.set_column(i, output);
//	}
//
//	if (constructed_RobustKernelPCA.cols() != n || constructed_RobustKernelPCA.rows() != m_numberOfLandmarks * 3)
//	{
//		std::cout << " dimensions (number of samples) do not match! " << std::endl;
//		return;
//	}
//
//	std::cout << __FUNCTION__ << " : Finish performRobustKPCA in [RobustKPCA] -> Got constructed_RobustKPCA " << std::endl;
//
//
//	//////////// Update m_dataMatrix with constructed_RobustKernelPCA //////////////////////////////////////////////////////
//	for (int i = 0; i < this->m_numberOfSamples; i++)
//	{
//		this->m_dataMatrix.set_column(i, this->constructed_RobustKernelPCA.get_column(i));
//	}
//
//	std::cout << __FUNCTION__ << " : Upgrade the training dataMatrix for back projection " << std::endl;
//
//	//////////////////////////////// Update m_meanShapeVector & m_meanShape ///////////////////////////////////////////////
//
//	this->m_meanShapeVector.fill(0.0);
//
//	for (int i = 0; i < this->m_numberOfSamples; i++)
//		this->m_meanShapeVector += this->m_dataMatrix.get_column(i);
//
//	this->m_meanShapeVector /= this->m_numberOfSamples;
//
//	cout << __FUNCTION__ << " End ... " << endl; 
//
//}
//

/**********************************************************************************************/


void RobustKPCA::performPCA()
{
	this->m_modelType = "PCA"; 

	this->constructed_PCA.set_size(this->m_dataMatrix.rows(), this->m_dataMatrix.cols());

	std::vector< pdm::Sample* >* samples_PCA = new std::vector< pdm::Sample* >;

	for (int n = 0; n < this->m_numberOfSamples; n++)
	{
		pdm::Sample* sample = new pdm::Sample(this->m_numberOfLandmarks, 3);

		for (int Cnt = 0; Cnt < this->m_dataMatrix.rows(); Cnt += 3)
		{
			double point[3];
			point[0] = this->m_dataMatrix[Cnt][n];
			point[1] = this->m_dataMatrix[Cnt + 1][n];
			point[2] = this->m_dataMatrix[Cnt + 2][n];

			sample->SetLandmark(Cnt / 3, point);
		}

		samples_PCA->push_back(sample);
	}

	if (this->m_Utils) delete this->m_Utils;

	this->m_Utils = new StatisticalUtilities();
	this->m_Utils->createAtlas(samples_PCA);

	pdm::ModelAbstract* modelPCA = new pdm::Model();
	modelPCA = this->m_Utils->GetModel();

	this->m_eigenValsReduced = dynamic_cast< pdm::Model* >(modelPCA)->GetEigenvalues();
	this->m_eigenVectorsReduced = dynamic_cast<pdm::Model*>(modelPCA)->GetEigenvectorMatrix();
	this->m_numberOfRetainedPCs = dynamic_cast<pdm::Model*>(modelPCA)->GetParameterCount();
	this->m_meanShapeVector = dynamic_cast<pdm::Model*>(modelPCA)->GetMeanShape();

	if (this->m_numberOfRetainedPCs != this->m_eigenVectorsReduced.cols() || this->m_numberOfRetainedPCs != this->m_eigenValsReduced.size())
	{
		cout << " Size of variances is not equal in PCA reconstruction ! " << endl;
		return;
	}


	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	cout << endl;
	cout << endl;
	cout << __FUNCTION__ << " : " << this->m_eigenValsReduced.size() << " eigenvalues from PCA training : " << endl;
	for (int i = 0; i < this->m_eigenValsReduced.size(); i++)
		cout << this->m_eigenValsReduced[i] << " ";
	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


	// back projection 
	for (int n = 0; n < samples_PCA->size(); n++)
	{
		pdm::Sample* projectSample = new pdm::Sample(this->m_dataMatrix.get_column(n), 1);
		this->m_Utils->Backproject(projectSample);

		VectorOfDouble projectedVec = projectSample->GetSampleVector();
		this->constructed_PCA.set_column(n, projectedVec);
	}

	cout << __FUNCTION__ << " End ... "; 
	cout << endl; 
	cout << endl;

}


void RobustKPCA::RobustPCA()
{
	this->m_modelType = "RPCA"; 

	double m = m_dataMatrix.rows();
	double n = m_dataMatrix.cols();

	double lambda = 1.0 / sqrt(m);        // this factor is very important, as usually n << m 
	double tol = 1e-6;
	int maxIter = 500;

	MatrixOfDouble Y = m_dataMatrix; // can be tuned

	vnl_svd< double > svdY(Y);
	double norm2 = svdY.sigma_max();
	double normInf = Y.operator_inf_norm() / lambda;
	double normDual = std::fmax(norm2, normInf);

	Y = Y / normDual;

	MatrixOfDouble m_lowM(m, n, 0.0);    // return this 
	MatrixOfDouble m_sparseM(m, n, 0.0);

	double dNorm = m_dataMatrix.frobenius_norm();

	double mu = 1.25 / norm2;   
	double mu_bar = mu * 1e7;
	double rho = 1.6;

	int iter = 0;
	int total_svd = 0;
	double converged = false;
	double stopCriterion = 1.0;

	while (!converged)
	{
		iter++;

		/**********************************************************************************/
		vnl_svd< double > svd(m_dataMatrix - m_sparseM + (1.0 / mu) * Y);
		MatrixOfDouble diagS = svd.W();
		VectorOfDouble diagSVec(diagS.rows());

		for (int i = 0; i < diagS.rows(); i++)
			diagSVec[i] = diagS[i][i];


		int svp = 0;
		for (int j = 0; j < diagSVec.size(); j++)
		{
			if (diagSVec[j] > 1 / mu) 
				svp++;
		}

		/**********************************************************************************/

		// U^T * U = I 
		// V^T * V = I 
		
		MatrixOfDouble U = svd.U();
		MatrixOfDouble V = svd.V();
		MatrixOfDouble tempU(U.rows(), svp); tempU.fill(0.0);
		for (int u = 0; u < svp; u++)
			tempU.set_column(u, U.get_column(u));

		MatrixOfDouble tempVTrans(V.rows(), svp); tempVTrans.fill(0.0);
		for (int v = 0; v < svp; v++)
			tempVTrans.set_column(v, V.get_column(v));
		tempVTrans = tempVTrans.transpose();

		MatrixOfDouble tempW(svp, svp); tempW.fill(0.0);
		for (int w = 0; w < svp; w++)
		{
			tempW[w][w] = diagSVec[w] - 1 / mu;  // if not shrink singular values, results are not good 
		}

		m_lowM = tempU * tempW * tempVTrans;

		/**********************************************************************************/

		/**********************************************************************************/

		MatrixOfDouble temp_M = m_dataMatrix - m_lowM + (1.0 / mu) * Y;

		for (int eCol = 0; eCol < temp_M.cols(); eCol++)
		{
			for (int eRow = 0; eRow < temp_M.rows(); eRow++)
			{
				m_sparseM[eRow][eCol] = std::fmax(temp_M[eRow][eCol] - lambda / mu, 0.0)
					+ std::fmin(temp_M[eRow][eCol] + lambda / mu, 0.0);
			}
		}

		/**********************************************************************************/

		total_svd++;

		MatrixOfDouble Z = m_dataMatrix - m_lowM - m_sparseM;

		int test = 0;
		int sparse = 0; 
		for (int i = 0; i < Z.cols(); i++)
		{
			for (int j = 0; j < Z.rows(); j++)
			{
				if (abs(Z[j][i]) > 0)
					test++;

				if (abs(m_sparseM[i][j]) > 0)
					sparse++; 
			}
		}

		Y = Y + mu * Z;
		mu = std::fmin(mu * rho, mu_bar);

		/**********************************************************************************/

		/**********************************************************************************/
		stopCriterion = Z.frobenius_norm() / dNorm;

		if (stopCriterion < tol)
		{
			converged = true;

			int e = 0;
			for (int eCol = 0; eCol < m_sparseM.cols(); eCol++)
			{
				for (int eRow = 0; eRow < m_sparseM.rows(); eRow++)
				{
					if (abs(m_sparseM[eRow][eCol]) > 0)
						e++;
				}
			}

			std::cout << "stopCriterion < tol : ";
			std::cout << "#svd: " << total_svd << ", #entries in |sparse matrix|: " << e << ", stopCriterion: " << stopCriterion;
			std::cout << " ; Iteration: " << iter << ", svp = " << svp << ", mu = " << mu << std::endl;
		}

		if (!converged && iter >= maxIter)
		{
			std::cout << "Maximum iterations reached" << std::endl;
			converged = true;
		}


		/** >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> **/

		std::cout << std::setprecision(6) << "iteration " << iter << " with 1/mu = " << 1 / mu
				<< " with the sum is " << diagSVec.sum() << "; svp = " << svp
				<< " , sparse entries = " << sparse << " , Z = " << test << std::endl;
		
		/** >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> **/


	}// end while

	constructed_RobustPCA.set_size(m, n);
	constructed_RobustPCA = m_lowM;

	/**********************************************************************************/

	// >>> Construct RPCA model with m_lowM >>> //

	std::vector< pdm::Sample* >* samples_RPCA = new std::vector< pdm::Sample* >;

	for (int col = 0; col < this->m_numberOfSamples; col++)
	{
		pdm::Sample* sample = new pdm::Sample(this->m_numberOfLandmarks, 3);

		for (int row = 0; row < constructed_RobustPCA.rows(); row += 3)
		{
			double point[3];
			point[0] = constructed_RobustPCA[row][col];
			point[1] = constructed_RobustPCA[row + 1][col];
			point[2] = constructed_RobustPCA[row + 2][col];

			sample->SetLandmark(row / 3, point);
		}
		samples_RPCA->push_back(sample);
	}

	if (!this->m_Utils) this->m_Utils = new StatisticalUtilities();
	else
	{
		delete this->m_Utils;
		this->m_Utils = new StatisticalUtilities();
	}

	this->m_Utils->createAtlas(samples_RPCA);
	pdm::ModelAbstract* modelRPCA = new pdm::Model();
	modelRPCA = this->m_Utils->GetModel();

	this->m_eigenValsReduced = dynamic_cast< pdm::Model* >(modelRPCA)->GetEigenvalues();
	this->m_eigenVectorsReduced = dynamic_cast<pdm::Model*>(modelRPCA)->GetEigenvectorMatrix();
	this->m_numberOfRetainedPCs = dynamic_cast<pdm::Model*>(modelRPCA)->GetParameterCount();
	this->m_meanShapeVector = dynamic_cast<pdm::Model*>(modelRPCA)->GetMeanShape();

	if (this->m_numberOfRetainedPCs != this->m_eigenVectorsReduced.cols() || this->m_numberOfRetainedPCs != this->m_eigenValsReduced.size())
	{
		cout << "  Size of variances is not equal in PCA reconstruction ! " << endl;
		return;
	}


	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	cout << endl;
	cout << endl;
	cout << __FUNCTION__ << " RPCA training - " << this->m_eigenValsReduced.size() << " eigenvalues from RPCA training are preserved : " << endl;
	for (int i = 0; i < this->m_eigenValsReduced.size(); i++)
		cout << this->m_eigenValsReduced[i] << " ";
	cout << endl;
	cout << endl;
	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


	std::cout << __FUNCTION__ << " End ... " << std::endl;


}


/**********************************************************************************************/


RobustKPCA::VectorOfDouble RobustKPCA::GetBackProjection(VectorOfDouble parameter)
{
	VectorOfDouble sample(this->m_dataMatrix.rows(), 0.0);

	if (parameter.size() != this->m_numberOfRetainedPCs)
	{
		std::cout << __FUNCTION__ << "  WRONG dimension " << std::endl;
		return sample;
	}

	if (this->m_modelType == "KPCA" || this->m_modelType == "Miccai17")
	{
		cout << __FUNCTION__ << " back project KPCA : " << endl; 
		sample = this->backProject(parameter);
	}
	else if (this->m_modelType == "NIPS09")   // compare with different kernel matrix 
	{
		cout << __FUNCTION__ << " back project NIPS09 : " << endl;  

		VectorOfDouble initShapeVector = this->backProject(parameter); 

		sample = this->backProjectNIPS09(initShapeVector, 0.1);
	}
	else if (this->m_modelType == "RKPCA")   // do the same thing 
	{
		cout << __FUNCTION__ << " back project robust : " << endl;

		VectorOfDouble inputShapeVector(this->m_dataMatrix.rows(), 0.0); 

		// find the similar kernel PCs 
		double minDist = std::numeric_limits< double >::max();
		for (int i = 0; i < this->m_numberOfSamples; i++)
		{
			double dist_i = vnl_vector_ssd(parameter, this->m_reducedKernelPCs.get_row(i)); 

			if (dist_i < minDist)
			{
				minDist = dist_i; 
				inputShapeVector = this->m_dataMatrix.get_column(i);
			}
		}

		sample = this->backProjectRobust(parameter, inputShapeVector);
	}
	else; 

	return sample;

}


vtkSmartPointer< vtkPolyData > RobustKPCA::ProjectShapeNonlinearModel(double _proportion, vtkSmartPointer< vtkPolyData > inputShape, int _modes)
{
	vtkSmartPointer< vtkPolyData > copyInput = vtkSmartPointer< vtkPolyData >::New();
	copyInput->DeepCopy(inputShape);

	if (this->m_kernelPCs.cols() == 0)
	{
		std::cout << " ERROR Not a nonlinear model ! " << std::endl;
		return NULL;
	}

	// preserve input Shape 
	std::vector< vtkSmartPointer< vtkPolyData > > aligned;

	aligned.push_back(this->m_dataSetPoly.at(0));
	aligned.push_back(copyInput);


	if (this->m_scaling)
	{
		this->AlignDataSetsWithScaling(aligned);
	}
	else
	{
		this->AlignDataSetsWithoutScaling(aligned);

//		this->CenterOfMassToOrigin(aligned); 
	}


	// preserve transformation 
	vtkSmartPointer< vtkLandmarkTransform > landmarkTransform = vtkSmartPointer< vtkLandmarkTransform >::New();
	landmarkTransform->SetSourceLandmarks(aligned.at(1)->GetPoints());
	landmarkTransform->SetTargetLandmarks(inputShape->GetPoints());
	if (this->m_scaling)
	{
		landmarkTransform->SetModeToSimilarity();
	}
	else
	{
		landmarkTransform->SetModeToRigidBody();
	}
	landmarkTransform->Update();


	/********************************************************************/

	if (_proportion > 0)
	{
		// remove missing entries 
		this->RemoveEntries(_proportion, aligned.at(1));
	}

	// convert to vector 
	VectorOfDouble shapeVector(aligned.at(1)->GetNumberOfPoints() * 3, 0.0);
	for (int i = 0; i < aligned.at(1)->GetNumberOfPoints() * 3; i += 3)
	{
		shapeVector[i] = aligned.at(1)->GetPoint(i / 3)[0];
		shapeVector[i + 1] = aligned.at(1)->GetPoint(i / 3)[1];
		shapeVector[i + 2] = aligned.at(1)->GetPoint(i / 3)[2];
	}

	//**************************************************************************************************************
	
	VectorOfDouble outputVector = shapeVector;

	// calculate the gram values with training database 
	VectorOfDouble gramVector(this->m_numberOfSamples, 0.0);

	for (int i = 0; i < this->m_numberOfSamples; i++)
	{
		double dist = vnl_vector_ssd(shapeVector, this->m_dataMatrix.get_column(i));
		gramVector[i] = exp(-dist / (2 * this->m_gamma * this->m_gamma));
	}

	VectorOfDouble inputKPCs = gramVector * this->m_eigenVectorsReduced;

	if (_modes >= 0 && _modes < this->m_numberOfRetainedPCs)
	{
		cout << __FUNCTION__ << " : projection with " << _modes << " preserved ... " << endl; 

		for (int cnt = _modes; cnt < this->m_numberOfRetainedPCs; cnt++)
			inputKPCs[cnt] = this->m_kPCsMean[cnt]; 
	}

	if (this->m_modelType == "KPCA" || this->m_modelType == "Miccai17")
	{
		outputVector = this->backProject(inputKPCs);
	}

	else if (this->m_modelType == "NIPS09")
	{
		outputVector = this->backProjectNIPS09(shapeVector, 0.1); 
	}

	else if (this->m_modelType == "RKPCA")
	{
		outputVector = this->backProjectRobust(inputKPCs, shapeVector);
	}

	else;

	//****************************************************************************************************************

	vtkPoints* points = aligned.at(1)->GetPoints();
	for (int i = 0; i < aligned.at(1)->GetNumberOfPoints() * 3; i += 3)
	{
		points->SetPoint(i / 3, outputVector[i], outputVector[i + 1], outputVector[i + 2]);
	}
	points->Modified();
	aligned.at(1)->Modified();


	// transform Back 
	vtkSmartPointer< vtkTransformPolyDataFilter > transformFilter = vtkSmartPointer< vtkTransformPolyDataFilter >::New();
	transformFilter->SetInputData(aligned.at(1));
	transformFilter->SetTransform(landmarkTransform);
	transformFilter->Update();
	aligned.at(1)->SetPoints(transformFilter->GetOutput()->GetPoints());
	aligned.at(1)->Modified();

	return aligned.at(1);

}


vtkSmartPointer< vtkPolyData > RobustKPCA::ProjectShapeLinearModel(double _proportion, vtkSmartPointer< vtkPolyData > inputShape, int _modes)
{
	if (!this->m_Utils)
	{
		std::cout << " ERROR Not a linear model ! " << std::endl;
		return NULL;
	}

	vtkSmartPointer< vtkPolyData > outputShape = vtkSmartPointer< vtkPolyData >::New();
	outputShape->DeepCopy(inputShape);


	// preserve input Shape 
	std::vector< vtkSmartPointer< vtkPolyData > > aligned;

	aligned.push_back(this->m_dataSetPoly.at(0));
	aligned.push_back(outputShape);


	if (this->m_scaling)
	{
		this->AlignDataSetsWithScaling(aligned);
	}
	else
	{
		this->AlignDataSetsWithoutScaling(aligned);
	}


	//>>> preserve transformation 
	vtkSmartPointer< vtkLandmarkTransform > landmarkTransform = vtkSmartPointer< vtkLandmarkTransform >::New();
	landmarkTransform->SetSourceLandmarks(aligned.at(1)->GetPoints());
	landmarkTransform->SetTargetLandmarks(inputShape->GetPoints());
	if (this->m_scaling)
	{
		landmarkTransform->SetModeToSimilarity();
	}
	else
	{
		landmarkTransform->SetModeToRigidBody();
	}
	landmarkTransform->Update();


	/**********************************************************************/

	// removing entries 
	if (_proportion > 0)
	{
		this->RemoveEntries(_proportion, aligned.at(1));
	}

	// convert to vector 
	VectorOfDouble shapeVector(aligned.at(1)->GetNumberOfPoints() * 3, 0.0);
	for (int i = 0; i < aligned.at(1)->GetNumberOfPoints() * 3; i += 3)
	{
		shapeVector[i] = aligned.at(1)->GetPoint(i / 3)[0];
		shapeVector[i + 1] = aligned.at(1)->GetPoint(i / 3)[1];
		shapeVector[i + 2] = aligned.at(1)->GetPoint(i / 3)[2];
	}

	// convert to sample 
	pdm::Sample* shapeSample = new pdm::Sample(shapeVector, 1);

	VectorOfDouble outputVector = this->backProjectLinear(shapeVector, _modes);

	vtkPoints* points = aligned.at(1)->GetPoints();
	for (int i = 0; i < aligned.at(1)->GetNumberOfPoints() * 3; i += 3)
	{
		points->SetPoint(i / 3, outputVector[i], outputVector[i + 1], outputVector[i + 2]);
	}
	points->Modified();
	aligned.at(1)->Modified();


	// transform Back 
	vtkSmartPointer< vtkTransformPolyDataFilter > transformFilter = vtkSmartPointer< vtkTransformPolyDataFilter >::New();
	transformFilter->SetInputData(aligned.at(1));
	transformFilter->SetTransform(landmarkTransform);
	transformFilter->Update();
	aligned.at(1)->SetPoints(transformFilter->GetOutput()->GetPoints());
	aligned.at(1)->Modified();

	return aligned.at(1);

}


RobustKPCA::MatrixOfDouble RobustKPCA::LowRankModelingKernelMatrix(MatrixOfDouble _inputData, MatrixOfDouble _kernelMatrix, MatrixOfDouble & m_K)
{
	int m = this->m_dataMatrix.rows();
	int n = this->m_dataMatrix.cols();

	MatrixOfDouble constructedDataMatrix(m, n, 0.0);
	MatrixOfDouble subGradientMatrix(m, n, 0.0);

	MatrixOfDouble eigenVectors(n, n, 0.0);
	VectorOfDouble eigenValues(n, 0.0);


	/****** Settings *******/
	
	vnl_svd< double > svdY_K(_kernelMatrix);
	double lambda = 1.0 / sqrt(n);
	double norm2 = svdY_K.sigma_max();  // 1.131 
	double normInf = _kernelMatrix.operator_inf_norm() / lambda;

	MatrixOfDouble kernel_Y = _kernelMatrix / std::fmax(norm2, normInf);

	kernel_Y.fill(0.0); 

	double mu = 1.25 / norm2;
	double mu_max = mu * 1e6;
	double mu_rho = 1.6;

	/******************************/

	int iter = 0;
	int iter_max = 100;

	bool directReturn = false;

	while (iter < iter_max)
	{
		vnl_svd< double > svd(_kernelMatrix + (1.0 / mu) * kernel_Y);     // SVD : O(n^3)
		MatrixOfDouble diagS = svd.W();

		int svp = 0;
		for (int j = 0; j < diagS.rows(); j++)
		{
			if (diagS[j][j] > 1.0 / mu)
				svp++;
		}

		//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//		cout << __FUNCTION__ << " : iter " << iter << " - total svd = " << svd.W().size() << " : ";
//		for (int i = 0; i < diagS.rows(); i++)
//			cout << diagS[i][i] << " ";
//		cout << endl;
		//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

		if (svp == 0)
		{
			if (iter == 0) directReturn = true;
			break;
		}

		MatrixOfDouble U = svd.U();
		MatrixOfDouble V = svd.V();
		MatrixOfDouble tempU(U.rows(), svp); tempU.fill(0.0);
		for (int u = 0; u < svp; u++)
		{
			tempU.set_column(u, U.get_column(u));
			eigenVectors.set_column(u, U.get_column(u));
		}

		MatrixOfDouble tempVTrans(V.rows(), svp); tempVTrans.fill(0.0);
		for (int v = 0; v < svp; v++)
			tempVTrans.set_column(v, V.get_column(v));
		tempVTrans = tempVTrans.transpose();

		MatrixOfDouble tempW(svp, svp); tempW.fill(0.0);
		for (int w = 0; w < svp; w++)
		{
			tempW[w][w] = diagS[w][w] - 1.0 / mu;
			eigenValues[w] = tempW[w][w];
		}

		//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
		cout << __FUNCTION__ << " : iter " << iter << " - preserved svp = " << svp << " SVs with thres = (" << 1.0 / mu << ") : ";
		for (int i = 0; i < tempW.rows(); i++)
			cout << tempW[i][i] << " ";
		cout << endl;
		//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

		m_K = tempU * tempW * tempVTrans;

		/**************************************************************************************************/

		MatrixOfDouble Z_K = _kernelMatrix - m_K;

		kernel_Y = kernel_Y + mu * Z_K;
		mu = std::fmin(mu * mu_rho, mu_max);

		double stopCriterionK = Z_K.frobenius_norm() / _kernelMatrix.frobenius_norm();

		int z = 0;
		for (int eCol = 0; eCol < Z_K.cols(); eCol++)
		{
			for (int eRow = 0; eRow < Z_K.rows(); eRow++)
			{
				if (abs(Z_K[eRow][eCol] > 0))
					z++;
			}
		}

		if (stopCriterionK < 1e-6)
		{
			std::cout << "Converged true : " << z << std::endl;
			break;
		}

		iter++; 

	}

	if (directReturn) return _inputData;

//	constructedDataMatrix = this->computeGradientLowRankX(_inputData, m_K);

//	return constructedDataMatrix;


	subGradientMatrix = this->subgradientKernelMatrix(_inputData, m_K);    // O(mn)

	return subGradientMatrix;


}


/*******************************************************************************************/


RobustKPCA::VectorOfDouble RobustKPCA::backProject(VectorOfDouble input)
{
	if (input.size() != this->m_eigenVectorsReduced.cols())  // this->_evecSelected.cols() = numRetainedPCs
	{
		std::cout << " Caution : Wrong dimension in reduced eigenVectors ! ";
		return this->m_meanShapeVector;
	}
	
	unsigned int N = this->m_numberOfSamples;

	VectorOfDouble _gammas(this->m_numberOfRetainedPCs, 0.0);
	_gammas = this->m_eigenVectorsReduced * input;  // input : kernel principal components 

	VectorOfDouble Z(this->m_dataMatrix.rows(), 0.0);
	VectorOfDouble preImgZ(this->m_dataMatrix.rows(), 0.0);

	Z = this->m_meanShapeVector; 

	double tol = 1e-5;
	int iter = 0;

	while (iter < 1000)
	{
		preImgZ = Z;

		VectorOfDouble kzD(this->m_dataMatrix.cols(), 0.0);
		for (int i = 0; i < kzD.size(); i++)
		{
			double dist = vnl_vector_ssd(this->m_dataMatrix.get_column(i), Z);   //  O(mn) 
			kzD[i] = exp(-dist / (2 * this->m_gamma * this->m_gamma));
		}
	
		vnl_vector< double > sumGamms(N, 0.0);
		for (int i = 0; i < N; i++)
			sumGamms[i] = kzD[i] * _gammas[i];   // O(n)

		double sumXXGamms = sumGamms.sum();

		if (sumXXGamms == 0)
		{
			// check kPC constraint again 
			VectorOfDouble newKPCs = kzD * this->m_eigenVectorsReduced;

			for (int i = 0; i < newKPCs.size(); i++)
			{
				double constraint_min = this->m_reducedKernelPCs.get_column(i).min_value();
				double constraint_max = this->m_reducedKernelPCs.get_column(i).max_value();

				if (newKPCs[i] > constraint_max)
				{
//					cout << " exceed max in " << i << " -th dimension " ;
					newKPCs[i] = constraint_max;
				}
					
				
				if (newKPCs[i] < constraint_min)
				{
//					cout << " exceed min in " << i << " -th dimension " ;
					newKPCs[i] = constraint_min;
				}
			}
		
			_gammas = this->m_eigenVectorsReduced * newKPCs;  // input : kernel principal components 

			Z = this->m_meanShapeVector;

			iter++; 

			continue; 
			
		}

		
		VectorOfDouble output(this->m_dataMatrix.rows(), 0.0);
		output = sumGamms * this->m_dataMatrix.transpose();     // 1 * n * n * m = 1 * n * m ?? 

		output /= sumXXGamms;
		Z = output;


		// reconstruction converged ! no more change of reconstruction 
		if ((preImgZ - Z).two_norm() / Z.two_norm() < tol)
		{
//			cout << " iteration < tol = " << iter << " with sumGamms = " << sumXXGamms << endl;
			return Z;
		}

		iter++;

	}

//	std::cout << "maximum iteration has been reached!" << std::endl;

	return Z;
}


RobustKPCA::VectorOfDouble RobustKPCA::backProjectRobust(VectorOfDouble input, VectorOfDouble shapeVector)
{
	cout << __FUNCTION__ << " : ";

	/**********************************************************************/

	if (input.size() != this->m_eigenVectorsReduced.cols())  // this->_evecSelected.cols() = numRetainedPCs
	{
		std::cout << " Caution : Wrong dimension in reduced eigenVectors ! ";
		return this->m_meanShapeVector;
	}

	unsigned int N = this->m_numberOfSamples;

	VectorOfDouble _gammas(this->m_numberOfRetainedPCs, 0.0);
	_gammas = this->m_eigenVectorsReduced * input;  // kernel principal components


	VectorOfDouble Z(this->m_dataMatrix.rows(), 0.0);
	VectorOfDouble preImgZ(this->m_dataMatrix.rows(), 0.0);

	double minDist = std::numeric_limits< double > ::max(); 
	int order = this->m_dataMatrix.cols();
	for (int i = 0; i < m_dataMatrix.cols(); i++)
	{
		double dist_i = sqrt(vnl_vector_ssd(shapeVector, this->m_dataMatrix.get_column(i)));
		if (dist_i < minDist)
		{
			minDist = dist_i;
			order = i;
		}
	}
	VectorOfDouble ClusterShape = this->m_dataMatrix.get_column(order);

	double distWithMean = sqrt(vnl_vector_ssd(shapeVector, this->m_meanShapeVector));
	if (minDist > distWithMean)
	{
		cout << __FUNCTION__ << " pass the mean shape vector as initialization ... " << endl; 
		Z = this->m_meanShapeVector; 
	}
	else
	{
		cout << __FUNCTION__ << " cluster i = " << order << endl; 
		Z = this->m_dataMatrix.get_column(order); 
	}

	//**>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

	double tol = 1e-7;
	int iter = 0;
	double lambda = 1 / (this->m_gamma * this->m_gamma);
	double constant = 1 / this->m_numberOfSamples;

	double omega1 = 1.0;
	double omega2 = 1.0;

	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


	double eta = 0.01;

	while (iter < 1000)
	{
		preImgZ = Z;

		VectorOfDouble kernelDistWithM(this->m_dataMatrix.cols(), 0.0);
		for (int i = 0; i < kernelDistWithM.size(); i++)
		{
			double dist = vnl_vector_ssd(this->m_dataMatrix.get_column(i), Z);
			kernelDistWithM[i] = exp(-dist / (2 * this->m_gamma * this->m_gamma));
		}

		vnl_vector< double > sumGammsBase(N, 0.0);
		sumGammsBase = element_product(kernelDistWithM, _gammas);  // kernelDistWithM (initial) = K_i 

		//vnl_vector< double > sumGamms(N, 0.0);
		//for (int i = 0; i < N; i++)
		//{
		//	sumGamms[i] = kernelDistWithM[i] * _gammas[i] - 2 * lambda * constant; 
		//}


		//double sumXXGamms = sumGamms.sum();

		//if (sumXXGamms == 0)   // does not find any similar cluster ， sum_kernelDistWithM = 0 
		//{
		//	cout << " => sumXXGamms = 0 in " << iter << " iteration ";

		//	// check kPC constraint again 
		//	VectorOfDouble newKPCs = kernelDistWithM * this->m_eigenVectorsReduced;

		//	for (int i = 0; i < newKPCs.size(); i++)
		//	{
		//		double constraint_min = this->m_reducedKernelPCs.get_column(i).min_value();
		//		double constraint_max = this->m_reducedKernelPCs.get_column(i).max_value();

		//		if (newKPCs[i] > constraint_max)
		//		{
		//			cout << " exceed max in " << i << " -th dimension ";

		//			newKPCs[i] = constraint_max;
		//		}


		//		if (newKPCs[i] < constraint_min)
		//		{
		//			cout << " exceed min in " << i << " -th dimension ";

		//			newKPCs[i] = constraint_min;
		//		}
		//	}

		//	cout << endl;

		//	_gammas = this->m_eigenVectorsReduced * newKPCs;  // input : kernel principal components 

		//	Z = this->m_meanShapeVector;

		//	iter++;

		//	continue;
		//}

		MatrixOfDouble diff(this->m_dataMatrix.rows(), this->m_dataMatrix.cols(), 0.0); 
		for (int i = 0; i < this->m_dataMatrix.cols(); i++)
		{
			diff.set_column(i, Z - this->m_dataMatrix.get_column(i)); 
		}

		
		VectorOfDouble output(this->m_dataMatrix.rows(), 0.0);
//		output = sumGammsBase * this->m_dataMatrix.transpose();   // recover later 
		output = sumGammsBase * diff.transpose(); 

		// kernel With ClusteShape; 
		double dist_Cluster = vnl_vector_ssd(Z, ClusterShape); 
		dist_Cluster = exp(- dist_Cluster / (2 * this->m_gamma * this->m_gamma)); 

		VectorOfDouble distanceCluster = dist_Cluster * (Z - ClusterShape); 
		

		//for (int i = 0; i < shapeVector.size(); i++)  // recover  
		//{
		//	output[i] = output[i] - lambda * constant * shapeVector[i] - lambda * constant * ClusterShape[i];   // new added 
		//}

		output = output - omega2 * distanceCluster;

		Z = preImgZ + eta * output;


//		output /= sumXXGamms;  // recover later 
//		Z = output;

		
		// reconstruction converged ! no more change of reconstruction 
		if ((preImgZ - Z).two_norm() / Z.two_norm() < tol)
		{
//			cout << " iteration < tol = " << iter << " with eta = " << eta << endl;
			return Z;
		}

		eta /= 10;

		iter++;

	}

	std::cout << "maximum iteration has been reached!" << std::endl;

	return Z;

}


RobustKPCA::VectorOfDouble RobustKPCA::backProjectAwesome(VectorOfDouble _inputKPCs, VectorOfDouble _inputShape, double _constBP)
{
	int m = this->m_dataMatrix.rows();
	int n = this->m_numberOfSamples;

	// mK : mean of Kernel Matrix 
	VectorOfDouble meanK(n, 0.0);
	for (int i = 0; i < n; i++)
		meanK += this->m_gramMatrix.get_column(i);
	meanK /= n;

	VectorOfDouble meanGammas = this->m_eigenVectorsReduced.transpose() * meanK;  // gammas: [k * 1] = [k * n * n * 1]

	//==================================================================

	VectorOfDouble _gammas(this->m_numberOfRetainedPCs, 0.0);
	_gammas = this->m_eigenVectorsReduced * _inputKPCs;  // input : kernel principal components 

	VectorOfDouble preImgZ(m, 0.0);

	//================================================================

	VectorOfDouble Z = _inputShape;  // x 

	double Const = 0.1;
	double tol1 = 1e-5;
	double tol2 = 1e-10;

	int iter = 0;
	while (iter < 300)  // for bad reconstruction, convergence easily 
	{		
		preImgZ = Z;

		//==================================================================================

		double dist = vnl_vector_ssd(_inputShape, Z);
		double kzx = exp(-dist / (2 * this->m_gamma * this->m_gamma));

		VectorOfDouble kzD(n, 0.0);    // n x 1 
		for (int i = 0; i < n; i++)
		{
			double dist = vnl_vector_ssd(this->m_dataMatrix.get_column(i), Z);
			kzD[i] = exp(-dist / (2 * this->m_gamma * this->m_gamma));
		}


		VectorOfDouble sumGamms(n, 0.0);
		for (int i = 0; i < n; i++)
			sumGamms[i] = kzD[i] * _gammas[i];   // this->m_eigenVectorsReduced * _inputKPCs;

		double Denominator1 = sumGamms.sum();

		VectorOfDouble Numerator1 = sumGamms * this->m_dataMatrix.transpose();
		
		//===================================================================================

		VectorOfDouble A = this->m_eigenVectorsReduced.transpose() * kzD; // A : [k * 1] = [k * n] * [n * 1]

		MatrixOfDouble B_tmp(m, n, 0.0);   // dataMatrix .* kzD[i] 
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				B_tmp[i][j] *= kzD[j];
			}
		}
		MatrixOfDouble B = B_tmp * m_eigenVectorsReduced;  // B : [m * k] = [m * n] * [n * k]


		double GammasA = 0.0;
		for (int i = 0; i < A.size(); i++)
			GammasA += meanGammas[i] * A[i];

		double Denominator2 = kzx + Const * (kzD.sum() / n + A.two_norm() - GammasA); // * A.two_norm(): A.two_norm() is better than its square 

		VectorOfDouble Numerator2 = _inputShape * kzx + Const * (this->m_dataMatrix * kzD / n + B * A - B * meanGammas);
		
		//===================================================================================

		if ((1 - _constBP) * Denominator1 == 0 && _constBP* Denominator2 == 0)
		{
//			cout << " iter " << iter << " both Denominator = zero ! " << endl; 

			Z = _inputShape; 

			iter++; 

			continue; 
		}

		Z = ((1 - _constBP) * Numerator1 + _constBP * Numerator2) / ((1 - _constBP) * Denominator1 + _constBP * Denominator2); 

		//===================================================================================
	
		double diff_iter = sqrt(vnl_vector_ssd(Z, preImgZ));
	

		if ((preImgZ - Z).two_norm() / Z.two_norm() < tol1 && diff_iter < tol2 )
		{
//			cout << " Output iter " << iter << " : Denominator1 = " << Denominator1 << ", Denominator2 = " << Denominator2 << endl;
			return Z;
		}

		/*if (iter % 50 == 0)
		{
			cout << iter << " : Denominator1 = " << Denominator1 << ", Denominator2 = " << Denominator2 << " , kzx = " << kzx << ", kzD = " << kzD.sum() << ", GammasA = " << GammasA << endl;
		}*/

		iter++;
	}

	return Z;
}


RobustKPCA::VectorOfDouble RobustKPCA::backProjectNIPS09(VectorOfDouble _inputShape, double _constBP)
{
	int m = this->m_dataMatrix.rows(); 
	int n = this->m_numberOfSamples; 

	// mK : mean of Kernel Matrix 
	VectorOfDouble meanK(n, 0.0); 
	for (int i = 0; i < n; i++)
		meanK += this->m_gramMatrix.get_column(i); 
	meanK /= n; 

	// gammas = eigenVectors * kernelMatrix * eigenVectors or kernel Matrix * eigenVectors * eigenVectors.transpose()
	
	VectorOfDouble meanKPCs = this->m_eigenVectorsReduced.transpose() * meanK;  // gammas: [k * 1] = [k * n * n * 1]

	VectorOfDouble Z = _inputShape;  // x 

	double Const = 0.09;
	double tol = 1e-10; 

	int iter = 0; 
	while (iter < 200)  // for bad reconstruction, convergence easily 
	{
		double dist = vnl_vector_ssd(_inputShape, Z);
		double kzx = exp(-dist / (2 * this->m_gamma * this->m_gamma));

		VectorOfDouble kzD(n, 0.0);    // n x 1 
		for (int i = 0; i < n; i++)
		{
			double dist = vnl_vector_ssd(this->m_dataMatrix.get_column(i), Z); 
			kzD[i] = exp(-dist / (2 * this->m_gamma * this->m_gamma)); 
		}

		
		VectorOfDouble A = this->m_eigenVectorsReduced.transpose() * kzD; // A : [k * 1] = [k * n] * [n * 1]

		MatrixOfDouble B_tmp(m, n, 0.0); 
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				B_tmp[i][j] *= kzD[j]; 
			}
		}
		MatrixOfDouble B = B_tmp * m_eigenVectorsReduced;  // B : [m * k] = [m * n] * [n * k]

		// Numerator : [m * 1] -> gradient 
		VectorOfDouble Numerator = _inputShape * kzx + Const * (this->m_dataMatrix * kzD / n + B * A - B * meanKPCs);


		double GammasA = 0.0; 
		for (int i = 0; i < A.size(); i++)
			GammasA += meanKPCs[i] * A[i];
		
	
		double Denominator = kzx + Const * (kzD.sum() / n + A.two_norm()  - GammasA); // * A.two_norm(): A.two_norm() is better than its square 

		if (Denominator == 0)
		{
			cout << " !!! WARN : kzx = " << kzx << " , kzD.sum = " << kzD.sum() << ", GammasA = " << GammasA << endl;
			continue; 
		}

		double diff_iter = sqrt(vnl_vector_ssd(Z, Numerator / Denominator)); 

		Z = Numerator / Denominator;   // Numerator / Denominator = P\phi(z) 

		if (diff_iter < tol)
		{
//			cout << __FUNCTION__ << " => " << iter << " output (const): " << Const << endl;
//			cout << " Denominator = " << Denominator << " , kzx = " << kzx << ", kzD.sum = " << kzD.sum() << ", GammasA = " << GammasA << endl;

			break;
		}

		/****************************************************/

		VectorOfDouble V = this->m_eigenVectorsReduced.transpose() * (kzD - meanK); // V : [k * 1] = [k * n] * [n * 1]

		double R = -2 / n * kzD.sum() - std::pow(V.two_norm(), 2); 
		
		double dist_err = vnl_vector_ssd(_inputShape, Z); 
		double Err_iter = -2 * exp(-dist_err / (2 * this->m_gamma * this->m_gamma)) + Const * R; 

		/****************************************************/

		/*if (iter % 5 == 0)
		{
			cout << __FUNCTION__ << " => " << iter << " : print immediate parameters : " << endl;
			cout << " Denominator = " << Denominator << " , kzx = " << kzx << ", kzD.sum = " << kzD.sum() << ", GammasA = " << GammasA << ", Err_iter = " << Err_iter << endl;
		}*/

		iter++; 
	}

	return Z; 
}


RobustKPCA::VectorOfDouble RobustKPCA::backProjectLinear(VectorOfDouble shapeVector, int m_modes)
{
	VectorOfDouble bParameters = this->m_eigenVectorsReduced.transpose() * (shapeVector - this->m_meanShapeVector); 

	if (m_modes >= 0)
	{
		for (int i = m_modes; i < this->m_numberOfRetainedPCs; i++)
			bParameters[i] = 0; 
	}

	for (int i = 0; i < this->m_numberOfRetainedPCs; i++)
	{
		if (bParameters[i] > 3 * sqrt(this->m_eigenValsReduced[i]))
			bParameters[i] = 3 * sqrt(this->m_eigenValsReduced[i]); 

		if (bParameters[i] < - 3 * sqrt(this->m_eigenValsReduced[i]))
			bParameters[i] = - 3 * sqrt(this->m_eigenValsReduced[i]);
	}

	// recompute the projected shapes 
	VectorOfDouble projectedShapeVector = this->m_meanShapeVector + this->m_eigenVectorsReduced * bParameters; 

	return projectedShapeVector;

}


RobustKPCA::VectorOfDouble RobustKPCA::backProjectInternal(VectorOfDouble inputKPCs, MatrixOfDouble inputDataMatrix, MatrixOfDouble inputEigenVecs, VectorOfDouble initVector)
{
	int m = inputDataMatrix.rows(); 
	int n = inputDataMatrix.cols(); 

	if (inputKPCs.size() != inputEigenVecs.cols())  // this->_evecSelected.cols() = numRetainedPCs
	{
		std::cout << " Caution : Wrong dimension in reduced eigenVectors ! ";
		return this->m_meanShapeVector;
	}

	VectorOfDouble _gammas = inputEigenVecs * inputKPCs;         // inputKPCs = m_K [n x n] * inputEigenVecs [n x k];  

	VectorOfDouble mean(m, 0.0); 
	for (int i = 0; i < n; i++)
		mean += inputDataMatrix.get_column(i); 
	mean /= n; 

	VectorOfDouble Z = initVector;
	VectorOfDouble preImgZ = Z; 

	/******************************************************/

	double tol = 1e-5;
	int iter = 0;

	while (iter < 500)
	{
		preImgZ = Z;

		VectorOfDouble kzD(n, 0.0);
		for (int i = 0; i < kzD.size(); i++)
		{
			double dist = vnl_vector_ssd(inputDataMatrix.get_column(i), Z);   // diff from data Matrix 
			kzD[i] = exp(- dist / (2 * this->m_gamma * this->m_gamma));
		}


		vnl_vector< double > sumGamms(n, 0.0);
		for (int i = 0; i < n; i++)
			sumGamms[i] = kzD[i] * _gammas[i];

		double sumXXGamms = sumGamms.sum();

		if (sumXXGamms == 0)
		{
			Z = mean;

			iter++;

			continue;
		}


		VectorOfDouble output(m, 0.0);
		output = sumGamms * inputDataMatrix.transpose();

		output /= sumXXGamms;
		Z = output;


		// reconstruction converged ! no more change of reconstruction 
		if ((preImgZ - Z).two_norm() / Z.two_norm() < tol)
		{
//			cout << " iteration < tol = " << iter << " with sumGamms = " << sumXXGamms << endl;
			return Z;
		}

		iter++;
	}

	return Z;

}

                                    
/*******************************************************************************************/


RobustKPCA::MatrixOfDouble RobustKPCA::computeGradientLowRankX(MatrixOfDouble _inputData, MatrixOfDouble _kernelMatrix)
{
	MatrixOfDouble Gradient = _inputData; 

	for (int sample = 0; sample < _inputData.cols(); sample++)
	{
		VectorOfDouble kzD = _kernelMatrix.get_column(sample); 

		VectorOfDouble output = kzD * _inputData.transpose(); // 1 * n * n * m = O (nm) * n =  O ( m n^2 ) 

		if (kzD.sum() == 0)
		{
			cout << __FUNCTION__ << " kzD.sum = 0 at sample " << sample << ", set column unchanged .. "<< endl;

			continue; 
		}

		output /= kzD.sum(); 

		Gradient.set_column(sample, output); 
	}

	return Gradient; 

}


RobustKPCA::MatrixOfDouble RobustKPCA::subgradientKernelMatrix(MatrixOfDouble _inputData, MatrixOfDouble _kernelMatrix)
{
	MatrixOfDouble subGradient = _inputData;

	for (int sample = 0; sample < _inputData.cols(); sample++)
	{
		VectorOfDouble kzD = _kernelMatrix.get_column(sample); // 1 * n

		VectorOfDouble output = _inputData * kzD;   // kzD * _inputData.transpose();    // O(mn)

		subGradient.set_column(sample, output);
	}

	return subGradient;

}


/*******************************************************************************************/


void RobustKPCA::AlignEvalGTWithTrainingMatrix(std::vector< vtkSmartPointer<vtkPolyData>> &evalGTPolys)
{
	cout << __FUNCTION__ << " : " << evalGTPolys.size() << " GT polys are loaded for evaluation ... " ; 

	if (evalGTPolys.size() != this->m_numberOfSamples)
	{
		cout << " Error Number of GT Polys ! " << endl; 
		return; 
	}

	m_evalGTMatrix.set_size(this->m_numberOfLandmarks * 3, this->m_numberOfSamples); 

	for (int i = 0; i < evalGTPolys.size(); i++)
	{
		std::vector< vtkSmartPointer<vtkPolyData> > tmpAlignedMeshes; 
		tmpAlignedMeshes.push_back(this->m_dataSetPoly.at(i)); 
		tmpAlignedMeshes.push_back(evalGTPolys.at(i)); 

		if (this->m_alignment)
		{
			if (this->m_scaling == true)
			{
				this->AlignDataSetsWithScaling(tmpAlignedMeshes);
			}
			else
			{
//				this->AlignDataSetsWithoutScaling(tmpAlignedMeshes);

				this->CenterOfMassToOrigin(tmpAlignedMeshes); 
			}
		}

		for (unsigned int j = 0; j < tmpAlignedMeshes.at(1)->GetNumberOfPoints() * 3; j += 3)
		{
			double pos[3];
			pos[0] = tmpAlignedMeshes.at(1)->GetPoint(j / 3)[0];
			pos[1] = tmpAlignedMeshes.at(1)->GetPoint(j / 3)[1];
			pos[2] = tmpAlignedMeshes.at(1)->GetPoint(j / 3)[2];

			this->m_evalGTMatrix[j][i] = pos[0];
			this->m_evalGTMatrix[j + 1][i] = pos[1];
			this->m_evalGTMatrix[j + 2][i] = pos[2];
		}
	}

	cout << " done ." << endl; 

}


void RobustKPCA::ComputeLinearModelGSWithPreservedModes(int numMode, int numSampling, double& _specificity, VectorOfDouble & _specVec)
{
	cout << endl; 
	cout << __FUNCTION__ << " Start : " << numMode << " modes are preserved with each mode generating " << numSampling << " samples .. " << endl;

	VectorOfDouble constraint(numMode, 0.0);
	for (int i = 0; i < numMode; i++)
		constraint[i] = 3.0 * sqrt(m_eigenValsReduced[i]); 

	MatrixOfDouble SamplesMatrix(this->m_numberOfLandmarks * 3, numSampling, 0.0); 

	std::srand((int)time(0)); 
	for (int Cnt = 0; Cnt < numSampling; Cnt++)
	{
		VectorOfDouble parameter(this->m_numberOfRetainedPCs, 0.0); 

		for (int inMode = 0; inMode < numMode; inMode++)
		{
			double randInx = (double)rand() / (double)RAND_MAX; 
			parameter[inMode] = -constraint[inMode] + randInx * 2 * constraint[inMode]; 
		}

		SamplesMatrix.set_column(Cnt, this->m_meanShapeVector + this->m_eigenVectorsReduced * parameter); 
	}

	/**>  Iterate SamplesMatrix to find the neareat m_dataMatrix **/
	for (int cntSamples = 0; cntSamples < SamplesMatrix.cols(); cntSamples++)
	{
		double minDist = std::numeric_limits< double >::max(); 
		for (int cntTraining = 0; cntTraining < m_evalGTMatrix.cols(); cntTraining++)
		{
//			double temp = this->getEuclideanDistanceOfVectors(SamplesMatrix.get_column(cntSamples), m_dataMatrix.get_column(cntTraining)); 
			
			double temp = this->getEuclideanDistanceOfVectors(this->m_evalGTMatrix.get_column(cntTraining), SamplesMatrix.get_column(cntSamples));
		
			if (temp < minDist) minDist = temp; 	
		}

		_specVec[cntSamples] = minDist; 
		_specificity += minDist; 
	}
	_specificity /= SamplesMatrix.cols(); 

	cout << __FUNCTION__ << " Specificity for " << numMode << " modes for linear-model is " << std::setprecision(6) << _specificity << endl;
	cout << __FUNCTION__ << " End ........ " << endl;

}


void RobustKPCA::ComputeNonlinearModelGSWithPreservedModes(int numMode, int numSampling, double& _specificity, VectorOfDouble & _specVec)
{
	cout << endl; 
	cout << __FUNCTION__ << " Start : " << numMode << " modes are preserved with each mode generating " << numSampling << endl;

	MatrixOfDouble constraint(numMode, 2, 0.0); 
	for (int i = 0; i < numMode; i++)
	{
		constraint[i][0] = this->m_kPCsMean[i] - 3 * this->m_kPCsStd[i];  // constraint[i] = sqrt(this->m_numberOfSamples * this->m_eigenValsReduced[i]);
		constraint[i][1] = this->m_kPCsMean[i] + 3 * this->m_kPCsStd[i]; 
	}
		

	MatrixOfDouble SamplesMatrix(this->m_numberOfLandmarks * 3, numSampling, 0.0); 

	std::srand((int)time(0));
	for (int Cnt = 0; Cnt < numSampling; Cnt++)
	{
		VectorOfDouble parameter(this->m_numberOfRetainedPCs, 0.0);
		for (int inMode = 0; inMode < numMode; inMode++)
		{
			double randInx = (double)rand() / (double)RAND_MAX;   // (0, 1)
//			parameter[inMode] = -constraint[inMode] + randInx * 2 * constraint[inMode];
			parameter[inMode] = constraint[inMode][0] + randInx * (constraint[inMode][1] - constraint[inMode][0]); 
		}

		SamplesMatrix.set_column(Cnt, this->GetBackProjection(parameter));
	}

	/********************************************************************************************/
	/**>  Iterate SamplesMatrix to find the neareat m_dataMatrix **/
	for (int cntSamples = 0; cntSamples < SamplesMatrix.cols(); cntSamples++)
	{
		double minDist = std::numeric_limits< double >::max();
		for (int cntTraining = 0; cntTraining < this->m_numberOfSamples; cntTraining++)
		{
			double temp = 1000.0; 
			
			temp = this->getEuclideanDistanceOfVectors(this->m_evalGTMatrix.get_column(cntTraining), SamplesMatrix.get_column(cntSamples));
			
			// temp = this->getEuclideanDistanceOfVectors(m_dataMatrix.get_column(cntTraining), SamplesMatrix.get_column(cntSamples));
			
			if (temp < minDist) minDist = temp;
		}
		_specVec[cntSamples] = minDist;
		_specificity += minDist;
	}
	_specificity /= SamplesMatrix.cols();


	cout << __FUNCTION__ << " Specificity for " << numMode << " modes for Nonlinear-model is " << std::setprecision(6) << _specificity << endl;
	cout << __FUNCTION__ << " End ........ " << endl;

}


double RobustKPCA::getEuclideanDistanceOfVectors(VectorOfDouble vec1, VectorOfDouble vec2)
{
	if (vec1.size() != vec2.size())
	{
		std::cout << " Wrong Dimension Match ! " << std::endl;
		return -1;
	}

	double distance = 0.0;
	for (int i = 0; i < vec1.size(); i += 3)
	{
		double temp = pow(vec1[i] - vec2[i], 2) + pow(vec1[i + 1] - vec2[i + 1], 2) + pow(vec1[i + 2] - vec2[i + 2], 2);

		distance += sqrt(temp);
	}

	distance /= vec1.size() / 3;

	return distance;
}


void RobustKPCA::RemoveEntries(double _proportion, vtkSmartPointer< vtkPolyData > & _mesh)
{
	int numRemoving = int(_proportion * this->m_numberOfLandmarks);

	cout << __FUNCTION__ << " removing " << numRemoving << " / " << this->m_numberOfLandmarks << " landmarks " << endl;

	std::srand((int)time(0));

	vtkPoints* points = _mesh->GetPoints();
	for (int Cnt = 0; Cnt < numRemoving; Cnt++)
	{
		double randInx = (double)rand() / (double)RAND_MAX;
		int removeOrder = int(randInx * this->m_numberOfLandmarks);

		if (removeOrder < 0) removeOrder = 0;
		if (removeOrder > this->m_numberOfLandmarks - 1) removeOrder = this->m_numberOfLandmarks - 1;

		points->SetPoint(removeOrder, 0.0, 0.0, 0.0);
	}
	points->Modified();
	_mesh->Modified();

	cout << __FUNCTION__ << " done. " << endl;
}


//>>>>> Perform Robust KPCA >>>>>//

//double lambda = 1.0 / n;  // 1.0 / sqrt((double)std::fmax(this->m_numberOfLandmarks, n));  // lambda only influence the sparse matrix computation, tuned
//double tol1 = 1e-10;
//double tol2 = 1e-5;
//int maxIter = 500;
//
//vnl_svd< double > svdK(this->m_gramMatrix);
//double norm2 = svdK.sigma_max();
//double normInf = this->m_gramMatrix.operator_inf_norm() / lambda;
//double normDual = std::fmax(norm2, normInf);
//double dNorm = this->m_gramMatrix.frobenius_norm();
//
//
//MatrixXd kernelM = MatrixXd::Zero(n, n);
//for (int i = 0; i < n; i++)
//{
//	for (int j = 0; j < n; j++)
//	{
//		kernelM(i, j) = this->m_gramMatrix[i][j];
//	}
//}
//
//
//MatrixXd m_L = MatrixXd::Zero(n, n);
//MatrixXd m_Y = MatrixXd::Zero(n, n);
//MatrixXd m_E = MatrixXd::Zero(n, n);
//MatrixXd m_I = MatrixXd::Identity(n, n);
//MatrixXd sqrtXTX = MatrixXd::Identity(n, n);
//MatrixXd m_L_old = MatrixXd::Zero(n, n);
//
//m_Y = kernelM / normDual;
//
//double mu = 1.25;
//double mu_max = mu * 1e6;
//double rho = 1.6;
//int iter = 0;
//double converged = false;
//
////==========================================================================================
//while (!converged)
//{
//	iter++;
//
//	////////////////////////////////////////////////////////////////////////
//
//	MatrixXd deltaE = kernelM - m_L + (1.0 / mu) * m_Y;
//
//	for (int eCol = 0; eCol < n; eCol++)
//	{
//		for (int eRow = 0; eRow < n; eRow++)
//		{
//			m_E(eRow, eCol) = std::fmax(deltaE(eRow, eCol) - 1 / mu, 0.0)
//				+ std::fmin(deltaE(eRow, eCol) + 1 / mu, 0.0);
//		}
//	}
//
//
//	////////////////////////////////////////////////////////////////////////////
//
//	MatrixXd tempA = mu * m_I;
//	arma::mat SylvesterA = zeros<mat>(n, n);
//	for (int i = 0; i < n; i++)
//	{
//		for (int j = 0; j < n; j++)
//			SylvesterA(i, j) = tempA(i, j);
//	}
//
//	MatrixXd temp209 = kernelM - m_E + mu * mu * m_I;  // avoid to be all zeros 
//	MatrixXd TempXTX = temp209.transpose() * temp209;
//
//	Eigen::EigenSolver< MatrixXd > eig(TempXTX);
//	MatrixXd eigVectors = eig.eigenvectors().real();
//	VectorXd eigValues = eig.eigenvalues().real();
//	MatrixXd eigVectorsInverse = eig.eigenvectors().real().inverse();
//
//	MatrixXd sqrtEigValues = MatrixXd::Zero(n, n);
//	for (int i = 0; i < n; i++)
//	{
//		if (eigValues(i) >= 0)
//			sqrtEigValues(i, i) = sqrt(eigValues(i));
//	}
//
//	sqrtXTX = eigVectors * sqrtEigValues * eigVectorsInverse;
//
//	//////////////////////////////////////////////////
//
//	MatrixXd tempB = sqrtXTX.inverse();
//	arma::mat SylvesterB = zeros<mat>(n, n);
//	for (int i = 0; i < n; i++)
//	{
//		for (int j = 0; j < n; j++)
//			SylvesterB(i, j) = tempB(i, j);
//	}
//
//	MatrixXd deltaL = kernelM - m_E + (1.0 / mu) * m_Y;
//	MatrixXd tempC = (-1) * mu * deltaL;
//	arma::mat SylvesterC = zeros<mat>(n, n);
//	for (int i = 0; i < n; i++)
//	{
//		for (int j = 0; j < n; j++)
//			SylvesterC(i, j) = tempC(i, j);
//	}
//
//	arma::mat X = zeros<mat>(n, n);
//	X = arma::syl(SylvesterA, SylvesterB, SylvesterC);
//
//	// convert to L 
//	for (int i = 0; i < n; i++)
//	{
//		for (int j = 0; j < n; j++)
//			m_L(i, j) = X(i, j);
//	}
//
//	//////////////////////////////////////////////////////////////////////////////
//
//	// m_L = kernelM - m_E + (1.0 / mu) * m_Y;
//	Eigen::JacobiSVD< MatrixXd > svd(m_L, ComputeThinU | ComputeThinV);
//
//	VectorXd singularValues = svd.singularValues();
//
//	int svp = 0;
//	for (int i = 0; i < n; i++)
//	{
//	if (singularValues(i) > 1.0 / mu)
//	svp++;
//	}
//
//	MatrixXd Sigmas = MatrixXd::Zero(svp, svp);
//	for (int i = 0; i < svp; i++)
//	{
//	Sigmas(i, i) = singularValues(i) - 1.0 / mu;
//	}
//
//	MatrixXd Left = MatrixXd::Zero(n, svp);
//	for (int i = 0; i < n; i++)
//	{
//	for (int j = 0; j < svp; j++)
//	Left(i, j) = svd.matrixU()(i, j);
//
//	}
//	MatrixXd Right = MatrixXd::Zero(n, svp);
//	for (int i = 0; i < n; i++)
//	{
//	for (int j = 0; j < svp; j++)
//	Right(i, j) = svd.matrixV()(i, j);
//	}
//
//	m_L = Left * Sigmas * Right.transpose();
//
//	//////////////////////////////////////////////////////////////////////////////
//
//	MatrixXd deltaK = kernelM - m_L - m_E;
//
//	int test = 0;
//	for (int i = 0; i < deltaK.cols(); i++)
//	{
//		for (int j = 0; j < deltaK.rows(); j++)
//		{
//			if (abs(deltaK(j, i)) <= 0)
//				test++;
//		}
//	}
//	std::cout << "iter = " << iter << ", deltaK = " << test << " , mu = " << mu << " , ";
//
//	m_Y = m_Y + mu * deltaK;
//	mu = std::fmin(mu * rho, mu_max);
//
//	MatrixOfDouble Z(n, n, 0.0);
//	for (int i = 0; i < n; i++)
//	{
//		for (int j = 0; j < n; j++)
//			Z[i][j] = deltaK(i, j);
//	}
//
//	double stopCriterion1 = Z.frobenius_norm() / dNorm;
//
//	if (!converged && iter >= maxIter)
//	{
//		std::cout << "Maximum iterations reached" << std::endl;
//		converged = true;
//	}
//
//
//	MatrixOfDouble DL(n, n, 0);
//	for (int i = 0; i < n; i++)
//	{
//		for (int j = 0; j < n; j++)
//			DL[i][j] = m_L_old(i, j) - m_L(i, j);
//	}
//
//
//	double stopCriterion2 = DL.frobenius_norm();
//
//	if (stopCriterion1 < tol1 && stopCriterion2 < tol2)
//	{
//		converged = true;
//		std::cout << "Both stopCriterions have been reached : " << stopCriterion2 << std::endl;
//		break;
//	}
//
//	m_L_old = m_L;
//
//	std::cout << " stop1 = " << stopCriterion1 << " , stop2 = " << stopCriterion2 << std::endl;
//
//}
//
////==========================================================================================
//
//cout << __FUNCTION__ << " L: " << endl;
//for (int i = 0; i < n; i++)
//{
//	for (int j = 0; j < n; j++)
//	{
//		this->m_gramMatrix[i][j] = m_L(i, j);
//		cout << m_L(i, j) << " ";
//	}
//	cout << endl;
//}
//cout << endl;
//
//
//MatrixOfDouble identity = MatrixOfDouble(n, n, 0.0);
//identity.set_identity();
//
//
//MatrixOfDouble Ones = MatrixOfDouble(n, n, 1.0);
//MatrixOfDouble H = identity - Ones / n;
//MatrixOfDouble kCenteredMatrix = H * this->m_gramMatrix * H;
//
//vnl_svd< double > svd(this->m_gramMatrix);
//this->m_eigenVectors = svd.U();
//this->m_eigenVals = svd.W().diagonal();
//
//double eps = 1e-05;  //5/2;
//for (unsigned int i = 0; i < n; i++)
//{
//	if (this->m_eigenVals[i] < eps)
//	{
//		this->m_eigenVals[i] = eps;
//		break;
//	}
//}
//
//// Normalize row such that for each eigenvector a with eigenvalue w we have w(a.a) = 1
//for (unsigned int i = 0; i < this->m_eigenVectors.cols(); i++)
//{
//	this->m_eigenVectors.scale_column(i, 1.0 / (sqrt(this->m_eigenVals[i])));
//}
//
//this->m_eigenVals /= static_cast<double>(n - 1);
//
//this->m_kernelPCs.set_size(n, n);
//this->m_kernelPCs = this->m_gramMatrix * this->m_eigenVectors;
//
//
//double cum = 0;
//for (unsigned int i = 0; i < this->m_eigenVals.size(); i++)
//{
//	cum += this->m_eigenVals.get(i);
//
//	if (cum / this->m_eigenVals.sum() >= 0.97)
//	{
//		this->m_numberOfRetainedPCs = i + 1; // count number
//		break;
//	}
//}
//
//this->m_eigenVectorsReduced.set_size(this->m_eigenVectors.rows(), this->m_numberOfRetainedPCs);
//this->m_eigenVectorsReduced = this->m_eigenVectors.extract(this->m_eigenVectors.rows(), this->m_numberOfRetainedPCs, 0, 0);
//
//this->m_eigenValsReduced.set_size(this->m_numberOfRetainedPCs);
//this->m_eigenValsReduced = this->m_eigenVals.extract(this->m_numberOfRetainedPCs, 0);
//
//
////>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//cout << endl;
//cout << __FUNCTION__ << " : RKPCA - reduced " << std::setprecision(6) << this->m_eigenValsReduced.size() << " eigenValues = " << endl;
//for (int i = 0; i < this->m_eigenValsReduced.size(); i++)
//	cout << this->m_eigenValsReduced[i] << " ";
//cout << endl;
//cout << endl;
////<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//
//
//this->m_reducedKernelPCs.set_size(n, this->m_numberOfRetainedPCs);
//this->m_reducedKernelPCs = this->m_gramMatrix * this->m_eigenVectorsReduced;  // row vectors stacked [ n * k ] 
//
//this->m_kPCsMean.set_size(this->m_numberOfRetainedPCs);
//this->m_kPCsStd.set_size(this->m_numberOfRetainedPCs);
//for (int i = 0; i < this->m_reducedKernelPCs.cols(); i++)
//{
//	VectorOfDouble tempKPCs = this->m_reducedKernelPCs.get_column(i);
//	this->m_kPCsMean[i] = tempKPCs.mean();
//
//	double dist = 0;
//	for (int j = 0; j < tempKPCs.size(); j++)
//	{
//		dist += pow(tempKPCs[j] - this->m_kPCsMean[i], 2);
//	}
//
//	this->m_kPCsStd[i] = sqrt(dist / tempKPCs.size());
//}
//
//
//this->constructed_RobustKernelPCA.set_size(this->m_dataMatrix.rows(), this->m_dataMatrix.cols());
//for (int i = 0; i < n; i++)
//{
//	VectorOfDouble output = this->backProjectRobust(this->m_reducedKernelPCs.get_row(i), this->m_dataMatrix.get_column(i));
//	this->constructed_RobustKernelPCA.set_column(i, output);
//}
//
//if (constructed_RobustKernelPCA.cols() != n || constructed_RobustKernelPCA.rows() != m_numberOfLandmarks * 3)
//{
//	std::cout << " dimensions (number of samples) do not match! " << std::endl;
//	return;
//}
//
//std::cout << __FUNCTION__ << " : Finish performRobustKPCA in [RobustKPCA] -> Got constructed_RobustKPCA " << std::endl;
//
//
////////////// Update m_dataMatrix with constructed_RobustKernelPCA //////////////////////////////////////////////////////
//for (int i = 0; i < this->m_numberOfSamples; i++)
//{
//	this->m_dataMatrix.set_column(i, this->constructed_RobustKernelPCA.get_column(i));
//}
//
//std::cout << __FUNCTION__ << " : Upgrade the training dataMatrix for back projection " << std::endl;
//
////////////////////////////////// Update m_meanShapeVector & m_meanShape ///////////////////////////////////////////////
//
//this->m_meanShapeVector.fill(0.0);
//
//for (int i = 0; i < this->m_numberOfSamples; i++)
//	this->m_meanShapeVector += this->m_dataMatrix.get_column(i);
//
//this->m_meanShapeVector /= this->m_numberOfSamples;
//
//cout << __FUNCTION__ << " End ... " << endl;
//



//>>>>>> Back Up Kernel low-ranking >>>>>//

//RobustKPCA::MatrixOfDouble RobustKPCA::LowRankModelingKernelMatrix(MatrixOfDouble _inputData, MatrixOfDouble _kernelMatrix, MatrixOfDouble & m_K)
//{
//	int m = this->m_dataMatrix.rows();
//	int n = this->m_dataMatrix.cols();
//
//	MatrixOfDouble constructedDataMatrix(m, n, 0.0);
//
//	MatrixOfDouble eigenVectors(n, n, 0.0);
//	VectorOfDouble eigenValues(n, 0.0);
//
//
//	/****** Settings *******/
//
//	MatrixOfDouble kernel_E(n, n, 0.0);
//	MatrixOfDouble kernel_Y = _kernelMatrix;
//
//	vnl_svd< double > svdY_K(kernel_Y);
//	double lambda = 1.0 / sqrt(n);
//	double norm2 = svdY_K.sigma_max();  // 1.131 
//	double normInf = kernel_Y.operator_inf_norm() / lambda;
//
//	//	kernel_Y = kernel_Y / std::fmax(norm2, normInf);
//
//	kernel_Y.fill(0.0);
//
//	double mu = 1.25 / norm2;
//	double mu_max = mu * 1e6;
//	double mu_rho = 1.6;
//
//	/******************************/
//
//	int iter = 0;
//	int iter_max = 100;
//
//	bool directReturn = false;
//
//	while (iter < iter_max)
//	{
//
//		/*****************************************************************/
//
//		//MatrixOfDouble temp_E = _kernelMatrix - m_K + (1.0 / mu) * kernel_Y;   // very initial kernel matrix 
//
//		//for (int eCol = 0; eCol < temp_E.cols(); eCol++)
//		//{
//		//	for (int eRow = 0; eRow < temp_E.rows(); eRow++)
//		//	{
//		//		kernel_E[eRow][eCol] = std::fmax(temp_E[eRow][eCol] - lambda / mu, 0.0)
//		//			+ std::fmin(temp_E[eRow][eCol] + lambda / mu, 0.0);
//		//	}
//		//}
//
//		/*********************************************************************************************/
//
//		vnl_svd< double > svd(_kernelMatrix - kernel_E + (1.0 / mu) * kernel_Y);
//		MatrixOfDouble diagS = svd.W();
//
//		int svp = 0;
//		for (int j = 0; j < diagS.rows(); j++)
//		{
//			if (diagS[j][j] > 1.0 / mu)
//				svp++;
//		}
//
//		//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//		cout << __FUNCTION__ << " : iter " << iter << " - total svd = " << svd.W().size() << " : ";
//		for (int i = 0; i < diagS.rows(); i++)
//			cout << diagS[i][i] << " ";
//		cout << endl;
//		//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//
//
//		if (svp == 0)
//		{
//			cout << __FUNCTION__ << " : iter " << iter << ", svp = 0. " << endl;
//
//			if (iter == 0) directReturn = true;
//
//			break;
//		}
//
//
//		MatrixOfDouble U = svd.U();
//		MatrixOfDouble V = svd.V();
//		MatrixOfDouble tempU(U.rows(), svp); tempU.fill(0.0);
//		for (int u = 0; u < svp; u++)
//		{
//			tempU.set_column(u, U.get_column(u));
//			eigenVectors.set_column(u, U.get_column(u));
//		}
//
//
//		MatrixOfDouble tempVTrans(V.rows(), svp); tempVTrans.fill(0.0);
//		for (int v = 0; v < svp; v++)
//			tempVTrans.set_column(v, V.get_column(v));
//		tempVTrans = tempVTrans.transpose();
//
//		MatrixOfDouble tempW(svp, svp); tempW.fill(0.0);
//		for (int w = 0; w < svp; w++)
//		{
//			tempW[w][w] = diagS[w][w] - 1.0 / mu;
//			eigenValues[w] = tempW[w][w];
//		}
//
//		//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//		cout << __FUNCTION__ << " : iter " << iter << " - preserved svp = " << svp << " SVs with thres = (" << 1.0 / mu << ") : ";
//		for (int i = 0; i < tempW.rows(); i++)
//			cout << tempW[i][i] << " ";
//		cout << endl;
//		//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//
//		m_K = tempU * tempW * tempVTrans;
//
//
//		/**************************************************************************************************/
//
//		MatrixOfDouble Z_K = _kernelMatrix - m_K - kernel_E;
//
//		kernel_Y = kernel_Y + mu * Z_K;
//		mu = std::fmin(mu * mu_rho, mu_max);
//
//		double stopCriterionK = Z_K.frobenius_norm() / _kernelMatrix.frobenius_norm();
//
//		int e = 0;
//		int z = 0;
//		for (int eCol = 0; eCol < kernel_E.cols(); eCol++)
//		{
//			for (int eRow = 0; eRow < kernel_E.rows(); eRow++)
//			{
//				if (abs(kernel_E[eRow][eCol]) > 0)
//					e++;
//
//				if (abs(Z_K[eRow][eCol] > 0))
//					z++;
//			}
//		}
//
//		if (stopCriterionK < 1e-8)
//		{
//			std::cout << "Converged true" << std::endl;
//			break;
//		}
//
//		iter++;
//
//	}
//
//	if (directReturn) return _inputData;
//
//	/////** Center Matrix H **/
//	////MatrixOfDouble identity = MatrixOfDouble(n, n, 0.0);
//	////identity.set_identity();
//
//	/////** centralizeKMatrix **/
//	////MatrixOfDouble Ones = MatrixOfDouble(n, n, 1.0);
//	////MatrixOfDouble H = identity - Ones / n;
//	////MatrixOfDouble kCenteredMatrix = H * m_K * H;
//
//	////MatrixOfDouble eigenVectors(n, n, 0.0);
//	////VectorOfDouble eigenValues(n, 0.0);
//	////this->eigenDecomposition(kCenteredMatrix, eigenVectors, eigenValues);
//
//	//double cum = 0;
//	//int numRetainedPCs = 0;
//	//for (unsigned int i = 0; i < eigenValues.size(); i++)  
//	//{
//	//	cum += eigenValues.get(i);
//
//	//	if (cum / eigenValues.sum() >= 0.95)   // remove all zeros 
//	//	{
//	//		numRetainedPCs = i + 1; 
//	//		break;
//	//	}
//	//}
//
//	//for (int i = 0; i < eigenVectors.cols(); i++)
//	//	eigenVectors.get_column(i).normalize(); 
//
//	//MatrixOfDouble redEigenVectors = eigenVectors.extract(eigenVectors.rows(), numRetainedPCs, 0, 0);
//	//VectorOfDouble redEigenValues = eigenValues.extract(numRetainedPCs, 0);
//	//MatrixOfDouble redKernelPCs = m_K * redEigenVectors;
//
//
//	////>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
//	//cout << endl;
//	//cout << __FUNCTION__ << " : " << redEigenValues.size() << " reduced eigenValues = ";
//	//for (int i = 0; i < redEigenValues.size(); i++)
//	//	cout << std::setprecision(3) << redEigenValues[i] << " ";
//	//cout << endl;
//	////<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//
//
//	/*for (int i = 0; i < n; i++)
//	{
//	VectorOfDouble output = this->backProjectInternal(redKernelPCs.get_row(i), _inputData, redEigenVectors, _inputData.get_column(i));
//
//	constructedDataMatrix.set_column(i, output);
//	}*/
//
//	constructedDataMatrix = this->computeGradientLowRankX(_inputData, m_K);
//
//	return constructedDataMatrix;
//
//}


