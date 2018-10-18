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

// Blueberry
#include <berryISelectionService.h>
#include <berryIWorkbenchWindow.h>

// Qmitk
#include "NonlinearModelling.h"


// Qt
#include <QMessageBox>


// MyAwesomeLib
#include <AwesomeImageFilter.h>


const std::string NonlinearModelling::VIEW_ID = "my.awesomeproject.views.awesomeview";


NonlinearModelling::NonlinearModelling()
	: m_robustKPCA(nullptr)
{
	m_recentDir = "J:\\Datasets";

	m_modelName = ""; 

	m_modes.fill(0.0); 
}


NonlinearModelling::~NonlinearModelling()
{
	if (m_robustKPCA != NULL)
		delete this->m_robustKPCA;
}


void NonlinearModelling::CreateQtPartControl(QWidget *parent)
{
	// create GUI widgets from the Qt Designer's .ui file
	m_Controls.setupUi(parent);

	connect(m_Controls.m_loadData, SIGNAL(clicked()), this, SLOT(LoadTrainingData()));
	connect(m_Controls.IncompleteDatasetsGenerator, SIGNAL(clicked()), this, SLOT(GenerateIncompleteDatasets())); 
	connect(m_Controls.LousyDatasetsGenerator, SIGNAL(clicked()), this, SLOT(GenerateLousyDatasets())); 
	connect(m_Controls.ResetDatasets, SIGNAL(clicked()), this, SLOT(ResetDatasets())); 

	connect(m_Controls.RPCA_ALM_Run, SIGNAL(clicked()), this, SLOT(RobustPCA()));
	connect(m_Controls.KPCA_Run, SIGNAL(clicked()), this, SLOT(KernelPCA()));
	connect(m_Controls.RKPCA_run, SIGNAL(clicked()), this, SLOT(RKPCA()));
	connect(m_Controls.MICCAI17_run, SIGNAL(clicked()), this, SLOT(Miccai17())); 
	
	connect(m_Controls.evalGS, SIGNAL(clicked()), this, SLOT(EvaluationGS()));
	connect(m_Controls.evalDistGT, SIGNAL(clicked()), this, SLOT(EvaluationDistGT()));
	connect(m_Controls.loadTestGTforBP, SIGNAL(clicked()), this, SLOT(LoadTestGTForBackProjection())); 
	connect(m_Controls.evalLoadTest, SIGNAL(clicked()), this, SLOT(LoadTestForBackProjection()));
	connect(m_Controls.evalLoadGT, SIGNAL(clicked()), this, SLOT(EvaluationLoadGTForCorruption())); 

	///////////////////////////////////////////////////////////////////////////////////////////////////////////

	m_Controls.m_alignment->setEnabled(true); 
	m_Controls.m_alignment->setChecked(true); 

	//*********************************************// 

	m_Controls.m_proportion->setEnabled(true);              // all training shapes are re-generated 
	m_Controls.m_proportion->setMaximum(1.00);
	m_Controls.m_proportion->setMinimum(0.00);
	m_Controls.m_proportion->setValue(0.00);

	m_Controls.m_numIncompleteDatasets->setEnabled(true);   // number of corrupted datasets 
	m_Controls.m_numIncompleteDatasets->setMaximum(1.00); 
	m_Controls.m_numIncompleteDatasets->setMinimum(0.00); 
	m_Controls.m_numIncompleteDatasets->setValue(1.00); 

	m_Controls.m_numRemovingPoints->setEnabled(true);       // number of removed landmarks 
	m_Controls.m_numRemovingPoints->setMaximum(5149); 
	m_Controls.m_numRemovingPoints->setMinimum(0); 
	m_Controls.m_numRemovingPoints->setSingleStep(1); 
	m_Controls.m_numRemovingPoints->setValue(1); 

	m_Controls.m_degreeRemoval->setEnabled(true);           // number of removed neighborhood vertices 
	m_Controls.m_degreeRemoval->setMaximum(1000); 
	m_Controls.m_degreeRemoval->setMinimum(0); 
	m_Controls.m_degreeRemoval->setValue(0); 

	m_Controls.m_startBasePoint->setEnabled(true);          // define the start point  
	m_Controls.m_startBasePoint->setMaximum(5149); 
	m_Controls.m_startBasePoint->setMinimum(-1); 
	m_Controls.m_startBasePoint->setValue(-1); 

	
	////////////////////////////////////////////////////////////////////////////////////////////////////////

	m_Controls.spinGamma->setEnabled(true);
	m_Controls.spinGamma->setValue(200);    
	m_Controls.spinGamma->setMaximum(50000);
	m_Controls.spinGamma->setMinimum(0.001); 

	m_Controls.spinNumSamples->setEnabled(true); 
	m_Controls.spinNumSamples->setMaximum(20000);
	m_Controls.spinNumSamples->setMinimum(10);
	m_Controls.spinNumSamples->setValue(500);

	m_Controls.spinNumRetainedModes->setEnabled(true); 
	m_Controls.spinNumRetainedModes->setValue(10);

	m_Controls.eval_proportion->setEnabled(true);
	m_Controls.eval_proportion->setMaximum(1.00); 
	m_Controls.eval_proportion->setMinimum(0.00); 
	m_Controls.eval_proportion->setValue(0.00); 

	m_Controls.mu_iter->setEnabled(true); 
	m_Controls.mu_iter->setMinimum(0); 
	m_Controls.mu_iter->setMaximum(500); 
	m_Controls.mu_iter->setValue(5); 

	m_Controls.printEvaluation->setEnabled(true); 
	m_Controls.printEvaluation->setText(QString("KidPair")); 
	
}


void NonlinearModelling::SetFocus()
{

}


//============================================ Load Datasets ================================================


void NonlinearModelling::LoadTrainingData()
{
	if (this->m_dataSetsBackup.size() > 0)
	{
		while (!m_dataSetsBackup.empty())
		{
			m_dataSetsBackup.pop_back();
		}
	}

	
	QFileDialog fd;
	fd.setDirectory(m_recentDir.c_str());
	fd.setFileMode(QFileDialog::Directory);
	QString selected;

	if (fd.exec() == QDialog::Accepted)
	{
		m_recentDir = fd.directory().absolutePath().toAscii();

		QStringList myfiles = fd.selectedFiles();

		if (!myfiles.isEmpty())
			selected = myfiles[0];
	}

	QDir* meshDir = new QDir(selected);

	QStringList::Iterator it;
	QStringList files = meshDir->entryList();
	it = files.begin();

	while (it != files.end())
	{
		if (QFileInfo(*meshDir, *it).isFile() && ((*it).endsWith)(".vtk"))
		{
			vtkSmartPointer< vtkPolyDataReader > pReader = vtkSmartPointer<vtkPolyDataReader>::New();

			pReader->SetFileName(QFileInfo(*meshDir, *it).absoluteFilePath().toAscii());
			std::string fileName = QFileInfo(*meshDir, *it).baseName().toStdString();
			pReader->Update();

			vtkSmartPointer< vtkPolyData > loadedPoly = pReader->GetOutput();
			this->ComputeNormals(loadedPoly); 

			m_dataSetsBackup.push_back(loadedPoly);
		}
		++it;
	}

	std::cout << m_dataSetsBackup.size() << " meshes are loaded from " << selected.toStdString() << std::endl;

	double sigma = 0.0; 
	this->estimateKernelWidth(m_dataSetsBackup, sigma); 

	this->VisualizeDatasets(m_dataSetsBackup, "gt", false); 

	delete meshDir;

	cout << __FUNCTION__ << " Start Training ! " << endl; 

}


void NonlinearModelling::GenerateIncompleteDatasets()
{
	cout << __FUNCTION__ << " Start Generating Incomplete Datasets ... " << endl; 

	if (m_dataSetsBackup.size() == 0)
	{
		MITK_WARN << " Please load training datasets ! " << endl; 
		return; 
	}

	if (this->m_corruptedDatasets.size() > 0)
	{
		while (!m_corruptedDatasets.empty())
		{
			m_corruptedDatasets.pop_back();
		}
	}

	for (int i = 0; i < m_dataSetsBackup.size(); i++)
	{
		vtkSmartPointer< vtkPolyData > poly = vtkSmartPointer< vtkPolyData >::New();
		poly->DeepCopy(m_dataSetsBackup.at(i));
		m_corruptedDatasets.push_back(poly);
	}

	// **************************************************************** //

	int M = this->m_dataSetsBackup.at(0)->GetNumberOfPoints(); 
	int N = this->m_dataSetsBackup.size(); 
	int numRemovingSamples = int(m_Controls.m_numIncompleteDatasets->value() * N);
	int numRemovingPoints = int(m_Controls.m_numRemovingPoints->value()); 

	if (m_Controls.m_startBasePoint->value() > 0) numRemovingPoints = 1; 
	

	// step 1: randomly select a shape to be made corrupted

	std::vector< int > samplesContainer;                    // avoid repetition 
	for (int i = 0; i < N; i++)
	{
		samplesContainer.push_back(i);
	}
	std::random_shuffle(samplesContainer.begin(), samplesContainer.end());

	std::vector< int > pointsContainer;                    // avoid repetition 
	for (int i = 0; i < M; i++)
	{
		pointsContainer.push_back(i);
	}

	for (int sampleCnt = 0; sampleCnt < numRemovingSamples; sampleCnt++)
	{
		int removedCnt = 0;
		int sampleOrder = samplesContainer[sampleCnt];
	
		std::random_shuffle(pointsContainer.begin(), pointsContainer.end());

		for (int pointCnt = 0; pointCnt < numRemovingPoints; pointCnt++)
		{
			std::vector< int > ids;

			// step 2: randomly select a base point to remove 

			int basePointID = pointsContainer[pointCnt];

			if (m_Controls.m_startBasePoint->value() >= 0) basePointID = m_Controls.m_startBasePoint->value(); 
			
			ids.push_back(basePointID);

			int baseIdsSize = ids.size();  

			vtkSmartPointer< vtkIdList > connectedVertices = this->GetConnectedVertices(m_corruptedDatasets.at(sampleOrder), basePointID);
			this->InsertIdsIntoArray(ids, connectedVertices);

			int updateIdsSize = ids.size();  

			for (int degree = 1; degree <= m_Controls.m_degreeRemoval->value(); degree++)
			{
				for (vtkIdType i = baseIdsSize; i < updateIdsSize; i++)
				{
					this->InsertIdsIntoArray(ids, this->GetConnectedVertices(m_corruptedDatasets.at(sampleOrder), ids.at(i)));
				}

				baseIdsSize = updateIdsSize; 

				updateIdsSize = ids.size();  
			}


			// step 4: set the selected points location to zero 

			vtkPoints* points = m_corruptedDatasets.at(sampleOrder)->GetPoints();

			for (int i = 0; i < ids.size(); i++)
			{
				points->SetPoint(ids.at(i), 0, 0, 0); 
			}

			points->Modified(); 
			m_corruptedDatasets.at(sampleOrder)->Modified();

			removedCnt += ids.size();
		}

		cout << __FUNCTION__ << " Sample " << sampleOrder << " , " << removedCnt << " points are removed .. " << endl;

	}

	double estimatedSigma = 0.0;
	this->estimateKernelWidth(m_corruptedDatasets, estimatedSigma);

	this->VisualizeDatasets(m_corruptedDatasets, "id", false);

	this->m_modelName = "id"; 
//	this->EvaluationDistGT(); 

	cout << __FUNCTION__ << " Incomplete Datasets Generator Done ...Start Training " << endl;

}


void NonlinearModelling::GenerateLousyDatasets()
{
	cout << __FUNCTION__ << " Generate Lousy Datasets ... " << endl; 

	if (m_dataSetsBackup.size() == 0)
	{
		MITK_WARN << " Please load training datasets ! " << endl;
		return;
	}

	if (this->m_corruptedDatasets.size() > 0)
	{
		while (!m_corruptedDatasets.empty())
		{
			m_corruptedDatasets.pop_back();
		}
	}

	for (int i = 0; i < m_dataSetsBackup.size(); i++)
	{
		vtkSmartPointer< vtkPolyData > poly = vtkSmartPointer< vtkPolyData >::New();
		poly->DeepCopy(m_dataSetsBackup.at(i));
		m_corruptedDatasets.push_back(poly);
	}

	// ***************************************************************************** // 

	int M = this->m_dataSetsBackup.at(0)->GetNumberOfPoints();
	int N = this->m_dataSetsBackup.size();
	int numRemovingPoints = int(m_Controls.m_proportion->value() * M);
	int numRemovingSamples = int(m_Controls.m_numIncompleteDatasets->value() * N);

	std::vector< int > pointsContainer;                     // avoid repetition 
	for (int i = 0; i < M; i++)
		pointsContainer.push_back(i); 
	
	std::vector< int > deformDegreeContainer;               // random selection 
	for (int i = 0; i < m_Controls.m_degreeRemoval->value(); i++)
		deformDegreeContainer.push_back(i); 


	for (int sample = 0; sample < numRemovingSamples; sample++)
	{
		std::random_shuffle(pointsContainer.begin(), pointsContainer.end());

		vtkDataArray* normalDeform = m_corruptedDatasets.at(sample)->GetPointData()->GetNormals();
		vtkPoints* points = m_corruptedDatasets.at(sample)->GetPoints();

		std::srand((int)time(0));
		for (int Cnt = 0; Cnt < numRemovingPoints; Cnt++)
		{
			int removeOrder = pointsContainer[Cnt]; 

			if (m_Controls.m_degreeRemoval->value() == 0)
			{
				points->SetPoint(removeOrder, 0, 0, 0);
			}
			else
			{
				double* location = points->GetPoint(removeOrder);

				// get normal of this landmark 
				double normalPoint[3];
				normalDeform->GetTuple(removeOrder, normalPoint);

				std::random_shuffle(deformDegreeContainer.begin(), deformDegreeContainer.end());

				double moveDirect[3]; 
				moveDirect[0] = -1 + 2 * (double)rand() / (double)RAND_MAX;
				moveDirect[1] = -1 + 2 * (double)rand() / (double)RAND_MAX; 
				moveDirect[2] = -1 + 2 * (double)rand() / (double)RAND_MAX; 

				double movePath[3];
		

				movePath[0] = normalPoint[0] * moveDirect[0] * deformDegreeContainer[0];
				movePath[1] = normalPoint[1] * moveDirect[1] * deformDegreeContainer[1];
				movePath[2] = normalPoint[2] * moveDirect[2] * deformDegreeContainer[2];
	
				points->SetPoint(removeOrder, location[0] + movePath[0], location[1] + movePath[1], location[2] + movePath[2]);

			}
		}

		points->Modified(); 
		m_corruptedDatasets.at(sample)->Modified(); 
	}

	double estimatedSigma = 0.0; 
	this->estimateKernelWidth(m_corruptedDatasets, estimatedSigma);

	this->VisualizeDatasets(m_corruptedDatasets, "ld", false);

	this->m_modelName = "ld";
	this->EvaluationDistGT();

	cout << __FUNCTION__ << " Generating Lousy Datasets Done ... Start Training " << endl;


}


void NonlinearModelling::ResetDatasets()
{
	if (m_dataSetsBackup.size() == 0)
	{
		MITK_WARN << " Please load training datasets ! " << endl;
		return;
	}

	if (this->m_corruptedDatasets.size() > 0)
	{
		while (!m_corruptedDatasets.empty())
		{
			m_corruptedDatasets.pop_back();
		}
	}

	for (int i = 0; i < m_dataSetsBackup.size(); i++)
	{
		vtkSmartPointer< vtkPolyData > poly = vtkSmartPointer< vtkPolyData >::New();
		poly->DeepCopy(m_dataSetsBackup.at(i));
		m_corruptedDatasets.push_back(poly);
	}

	double estimatedSigma = 0.0;
	this->estimateKernelWidth(m_corruptedDatasets, estimatedSigma);

	this->VisualizeDatasets(m_corruptedDatasets, "gt", false);

	cout << __FUNCTION__ << " Reset Done ... " << endl; 

}


//=============================================== Methods ================================================


/** Non-Linearity **/


void NonlinearModelling::KernelPCA()
{
	this->m_modelName = "KPCA"; 

	if (this->m_dataSetsBackup.size() == 0)
	{
		std::cout << " Please load datasets ahead! " << std::endl;
		return; 
	}

	/********************************************************************************************************/

	cout << endl; cout << __FUNCTION__ << " Start Training ... " << endl;

	if (this->m_robustKPCA)
	{
		cout << __FUNCTION__ << " Deleting the current model and create a new one ... " << endl; 
		delete this->m_robustKPCA;
	}

	m_robustKPCA = new RobustKPCA(m_Controls.spinGamma->value(), m_Controls.m_alignment->isChecked(), false);

	if (m_corruptedDatasets.size() == 0) m_robustKPCA->ReadDataMatrix(this->m_dataSetsBackup); 
	else m_robustKPCA->ReadDataMatrix(m_corruptedDatasets);

	itk::TimeProbe itkClock;
	itkClock.Start();

	m_robustKPCA->performKPCA();
	m_reconstructions = m_robustKPCA->getConstructed_KPCA();

	itkClock.Stop();
	MITK_INFO << "Time for Kernel PCA for " << m_reconstructions.at(0)->GetNumberOfPoints() << " points : " << std::setprecision(4) << itkClock.GetMean() << " seconds ." << std::endl;
	std::cout << std::endl;

	/********************************************************************************************************/

	m_Controls.spinNumRetainedModes->setValue(m_robustKPCA->GetParameterCount());

	cout << __FUNCTION__ << " End Training " << endl; cout << endl;

	this->EvaluationDistGT(); 
}


void NonlinearModelling::RKPCA()
{
	this->m_modelName = "RKPCA";

	if (this->m_dataSetsBackup.size() == 0)
	{
		std::cout << " Please load datasets ahead! " << std::endl;
		return;
	}
	
	/********************************************************************************************************/

	cout << endl;  cout << __FUNCTION__ << " Start Training ... " << endl;
	
	if (this->m_robustKPCA)  delete this->m_robustKPCA;
	
	m_robustKPCA = new RobustKPCA(m_Controls.spinGamma->value(), m_Controls.m_alignment->isChecked(), false);   // not scaling 

	if (m_corruptedDatasets.size() == 0) m_robustKPCA->ReadDataMatrix(this->m_dataSetsBackup);
	else m_robustKPCA->ReadDataMatrix(m_corruptedDatasets);

	itk::TimeProbe itkClock;
	itkClock.Start();

	m_robustKPCA->RKPCA(m_Controls.mu_iter->value(), m_Controls.m_proportion->value());
	m_reconstructions = m_robustKPCA->getConstructed_RKPCA();

	itkClock.Stop();
	MITK_INFO << "Time for RKPCA-18 " << m_reconstructions.at(0)->GetNumberOfPoints() << " points is " << std::setprecision(4) << itkClock.GetMean() << " seconds ." << std::endl;
	std::cout << std::endl;

	/********************************************************************************************************/

	m_Controls.spinNumRetainedModes->setValue(m_robustKPCA->GetParameterCount());

	cout << __FUNCTION__ << " End Training " << endl; cout << endl; 

	this->EvaluationDistGT();

}


void NonlinearModelling::Miccai17()
{
	this->m_modelName = "Miccai17";

	if (this->m_dataSetsBackup.size() == 0)
	{
		std::cout << " Please load datasets ahead! " << std::endl;
		return;
	}

	/********************************************************************************************************/

	cout << endl;  cout << __FUNCTION__ << " Start Training ... " << endl; 

	if (this->m_robustKPCA)
	{
		delete this->m_robustKPCA;
	}
	m_robustKPCA = new RobustKPCA(m_Controls.spinGamma->value(), m_Controls.m_alignment->isChecked(), false);   // not scaling 

	if (m_corruptedDatasets.size() == 0) m_robustKPCA->ReadDataMatrix(this->m_dataSetsBackup);
	else m_robustKPCA->ReadDataMatrix(m_corruptedDatasets);

	itk::TimeProbe itkClock;
	itkClock.Start();

	m_robustKPCA->Miccai17(); 
	m_reconstructions = m_robustKPCA->getConstructed_Miccai17(); 

	itkClock.Stop();
	MITK_INFO << "Time for Miccai-17 for " << m_reconstructions.at(0)->GetNumberOfPoints() << " landmarks is " << std::setprecision(4) << itkClock.GetMean() << " seconds ." << std::endl;
	std::cout << std::endl;

	/********************************************************************************************************/

	m_Controls.spinNumRetainedModes->setValue(m_robustKPCA->GetParameterCount());

	cout << __FUNCTION__ << " End Training .. " << endl; cout << endl;

	this->EvaluationDistGT();
}



/** Linearity **/


void NonlinearModelling::RobustPCA()
{
	this->m_modelName = "RPCA"; 

	if (this->m_dataSetsBackup.size() == 0)
	{
		std::cout << " Please load datasets ahead! " << std::endl;
		return;
	}

	/********************************************************************************************************/

	cout << endl; cout << __FUNCTION__ << " Start Training .. " << endl;

	if (this->m_robustKPCA) delete this->m_robustKPCA; 

	m_robustKPCA = new RobustKPCA(m_Controls.spinGamma->value(), m_Controls.m_alignment->isChecked(), false);
	
	if (m_corruptedDatasets.size() == 0) m_robustKPCA->ReadDataMatrix(this->m_dataSetsBackup);
	else m_robustKPCA->ReadDataMatrix(m_corruptedDatasets);

	itk::TimeProbe itkClock;
	itkClock.Start();

	m_robustKPCA->RobustPCA();
	m_reconstructions = m_robustKPCA->getConstructed_RobustPCA();

	itkClock.Stop();
	MITK_INFO << "Time for RobustPCA using ALM ( " << m_reconstructions.size() << " datasets ): " << itkClock.GetMean() << "seconds ." << std::endl;
	std::cout << std::endl;
	/********************************************************************************************************/

	m_Controls.spinNumRetainedModes->setValue(m_robustKPCA->GetParameterCount());

	cout << __FUNCTION__ << " End Training .. " << endl; cout << endl; 

	this->EvaluationDistGT();

}


//=============================================== Evaluation ==================================================================


void NonlinearModelling::EvaluationLoadGTForCorruption()
{
	cout << __FUNCTION__ << " Load Ground Truth Datasets for Evaluation ... ";

	if (this->m_evalGTDatasets.size() > 0)
	{
		while (!m_evalGTDatasets.empty())
		{
			m_evalGTDatasets.pop_back();
		}
	}

	QFileDialog fd;
	fd.setDirectory(m_recentDir.c_str());
	fd.setFileMode(QFileDialog::Directory);
	QString selected;

	if (fd.exec() == QDialog::Accepted)
	{
		m_recentDir = fd.directory().absolutePath().toAscii();
		QStringList myfiles = fd.selectedFiles();
		if (!myfiles.isEmpty())
			selected = myfiles[0];
	}

	QDir* meshDir = new QDir(selected);
	QStringList::Iterator it;
	QStringList files = meshDir->entryList();
	it = files.begin();

	while (it != files.end())
	{
		if (QFileInfo(*meshDir, *it).isFile() && ((*it).endsWith)(".vtk"))
		{
			vtkSmartPointer< vtkPolyDataReader > pReader = vtkSmartPointer< vtkPolyDataReader >::New();
			pReader->SetFileName(QFileInfo(*meshDir, *it).absoluteFilePath().toAscii());
			pReader->Update();

			vtkSmartPointer< vtkPolyData > loadedPoly = pReader->GetOutput(); // loaded poly to be projected
			m_evalGTDatasets.push_back(loadedPoly);
		}
		++it;
	}

	std::cout << m_evalGTDatasets.size() << " done ." << std::endl;

//	this->CenterOfMassToOrigin(m_evalGTDatasets, false); 

	this->VisualizeDatasets(m_evalGTDatasets, "GT", false); 


}


void NonlinearModelling::EvaluationGS()
{
	std::cout << std::endl;
	std::cout << "==================================================================== " << std::endl;
	std::cout << std::endl;

	if (!this->m_robustKPCA)
	{
		MITK_WARN << " Please train a model before ! "; 
		return; 
	}

	RobustKPCA::VectorOfDouble variances = this->m_robustKPCA->GetEigenValues();
	int numSampling = m_Controls.spinNumSamples->value(); 
	int preserveParamCount = std::min((int)variances.size(), m_Controls.spinNumRetainedModes->value());


	std::cout << __FUNCTION__ << " : Current " << this->m_modelName << " model has " << variances.size() << " modes of variance " << std::endl;
	std::cout << __FUNCTION__ << " : Preserve " << preserveParamCount << " modes to compute Generalization ability & Speciticity with " << numSampling << " samples generated . " << std::endl;
	std::cout << std::endl;

	std::string writeFileBaseName = m_Controls.printEvaluation->toPlainText().toStdString() + "_" + this->m_modelName; 
	std::stringstream ss_sigma; ss_sigma << m_Controls.spinGamma->value(); 

	writeFileBaseName += "_" + ss_sigma.str(); 

	/**************************************************************************************************/

	m_robustKPCA->AlignEvalGTWithTrainingMatrix(this->m_evalGTDatasets); 

	/**************************************************************************************************/

	RobustKPCA::VectorOfDouble specificity(preserveParamCount, 0.0);
	RobustKPCA::MatrixOfDouble specMatrix(numSampling, preserveParamCount, 0.0); 


	/**> Utilize visualization parameters **/
	if (this->m_modelName == "RPCA")
	{
		for (int i = 0; i < preserveParamCount; i++)
		{
			std::cout << __FUNCTION__ << " => for the " << i + 1 << " mode of model " << this->m_modelName << std::endl; 

			RobustKPCA::VectorOfDouble specVec(numSampling, 0.0); 
			this->m_robustKPCA->ComputeLinearModelGSWithPreservedModes(i + 1, numSampling, specificity[i], specVec);

			specMatrix.set_column(i, specVec); 
		}
	}

	else if (this->m_modelName == "KPCA" || this->m_modelName == "RKPCA" || this->m_modelName == "Miccai17")
	{
		for (int i = 0; i < preserveParamCount; i++)
		{
			std::cout << __FUNCTION__ << " => for the " << i + 1 << " mode of model " << this->m_modelName << std::endl;

			RobustKPCA::VectorOfDouble specVec(numSampling, 0.0);
			this->m_robustKPCA->ComputeNonlinearModelGSWithPreservedModes(i + 1, numSampling, specificity[i], specVec);

			specMatrix.set_column(i, specVec);
		}
	}

	/** save specificity to file **/
	std::ofstream _file_specificity;
	std::string fileSpecificity = writeFileBaseName + "_Specificity.txt";

	if (std::remove(fileSpecificity.c_str()) != 0) std::cout << " No such file! " << std::endl;
	else std::cout << " File " << fileSpecificity << " has been removed !" << std::endl;

	_file_specificity.open(fileSpecificity.c_str(), ios::app);

	if (!_file_specificity)
	{
		std::cout << " file cannot open! " << std::endl;
		return;
	}

	_file_specificity << this->m_modelName << "Specificity - the model has " << variances.size() << " modes with " << numSampling << " samples generated ... " << std::endl;


	double mean_S = specificity.mean();
	double SDS = 0.0;
	for (int i = 0; i < specificity.size(); i++)
	{
		_file_specificity << "Mode " << i + 1 << " : " << std::setprecision(6) << specificity[i] << std::endl;
		SDS += pow((specificity[i] - mean_S), 2);

		for (int sample = 0; sample < numSampling; sample++)
		{
			_file_specificity << specMatrix[sample][i] << endl;
		}
		_file_specificity << endl;
	}

	SDS /= specificity.size();
	SDS = sqrt(SDS);

	_file_specificity << "SD : " << std::setprecision(6) << SDS << std::endl;


	_file_specificity.close();

	std::cout << " end of evaluation of specificity with GT.  " << std::endl;


	/******************************************************************************************************/

	RobustKPCA::MatrixOfDouble generalizationLoO(this->m_dataSetsBackup.size(), m_Controls.spinNumRetainedModes->value(), 0.0);

	for (int sampleCnt = 0; sampleCnt < this->m_dataSetsBackup.size(); sampleCnt++)
	{
		std::vector< vtkSmartPointer< vtkPolyData >> trainPolySet; 
		for (int exclude = 0; exclude < this->m_dataSetsBackup.size(); exclude++)
		{
			if (exclude == sampleCnt) continue;
			
			trainPolySet.push_back(this->m_dataSetsBackup.at(exclude)); 
		}

		RobustKPCA::VectorOfDouble _generalization_sample(this->m_Controls.spinNumRetainedModes->value(), 0.0); 

		cout << "___________________________ Leave-" << sampleCnt << "-out_________________________________" << endl; 

		this->ComputeGeneralizationLoO(trainPolySet, this->m_dataSetsBackup.at(sampleCnt), this->m_evalGTDatasets.at(sampleCnt), _generalization_sample);

		generalizationLoO.set_row(sampleCnt, _generalization_sample); 

		cout << " LoO " << sampleCnt << " with generalization : "; 
		for (int i = 0; i < generalizationLoO.get_row(sampleCnt).size(); i++)
			cout << generalizationLoO.get_row(sampleCnt)[i] << " "; 
		cout << endl; 

		cout << endl; 
	}

	///******************************************************************************************************/

	
	/**> Save generalization to file **/
	std::ofstream _file_generalization;
	std::string fileGeneralization = writeFileBaseName + "_Generalization(LoO).txt";

	if (std::remove(fileGeneralization.c_str()) != 0) std::cout << " No such file! " << std::endl;
	else std::cout << " File " << fileGeneralization << " has been removed !" << std::endl;

	_file_generalization.open(fileGeneralization.c_str(), ios::app);

	if (!_file_generalization)
	{
		std::cout << " file cannot open! " << std::endl;
		return;
	}

	_file_generalization << this->m_modelName << "Generalization ability - the model preserves " << m_Controls.spinNumRetainedModes->value() << " modes ..." << std::endl;

	for (int i = 0; i < generalizationLoO.cols(); i++)
	{
		_file_generalization << "Mode " << i + 1 << " : " << std::setprecision(6) << generalizationLoO.get_column(i).mean() << std::endl;
		std::cout << "Mode " << i + 1 << " : " << std::setprecision(6) << generalizationLoO.get_column(i).mean() << std::endl;

		for (int row = 0; row < generalizationLoO.rows(); row++)
		{
			_file_generalization << generalizationLoO[row][i] << endl; 
		}
		_file_generalization << endl;
	}

	_file_generalization.close();

	std::cout << " end of evaluation of generalization ability with GT.  " << std::endl;


	/**************************************************************************************************/

	std::ofstream _file_compactness;
	std::string fileCompactness = writeFileBaseName + "_Compactness.txt";

	if (std::remove(fileCompactness.c_str()) != 0) std::cout << " No such file! " << std::endl;
	else std::cout << " File " << fileCompactness << " has been removed !" << std::endl;

	_file_compactness.open(fileCompactness.c_str(), ios::app);

	if (!_file_compactness)
	{
		std::cout << " file cannot open! " << std::endl;
		return;
	}

	_file_compactness << this->m_modelName << "Compactness - the model has " << variances.size() << " eigenvalues with " << numSampling << " samples generated ... " << std::endl;

	for (int i = 0; i < variances.size(); i++)
	{
		double compact_i = 0.0; 

		for (int j = 0; j <= i; j++)
		{
			compact_i += variances[j]; 
		}

		_file_compactness << std::setprecision(6) << compact_i / variances.sum() << endl;
	}

	_file_compactness.close(); 

	std::cout << " end of evaluation of Compactness with GT.  " << endl; 
	std::cout << std::endl; 
	std::cout << "============================================================= " << std::endl;
}


void NonlinearModelling::EvaluationDistGT()
{
	std::cout << std::endl;
	std::cout << "====================================================================" << std::endl;
	std::cout << std::endl;

	if (this->m_dataSetsBackup.size() == 0 && (this->m_reconstructions.size() == 0 || this->m_corruptedDatasets.size()))
	{
		MITK_WARN << " [ " << __FUNCTION__ << " ] Check DatasetsBackup && Reconstructions "; 
		return; 
	}

	std::string writeFileBaseName = m_Controls.printEvaluation->toPlainText().toStdString() + "_" + this->m_modelName;

	/***************************************************************************************/

	vnl_vector< double > reconstructionErrors(m_dataSetsBackup.size(), 0.0);
	for (int n = 0; n < m_dataSetsBackup.size(); n++)
	{
		vtkPolyData* inputPoly = m_dataSetsBackup.at(n);

		double dist_n = 0.0;

		if (this->m_reconstructions.size() == 0)
		{
			reconstructionErrors[n] = this->AlignedAndComputeEucliDistPolys(inputPoly, this->m_corruptedDatasets.at(n));
		}
		else
		{
			vtkPolyData* reconPoly = m_reconstructions.at(n);

			reconstructionErrors[n] = this->AlignedAndComputeEucliDistPolys(inputPoly, reconPoly);
		}	
	}


	/** Save reconstruction error to file **/

	std::stringstream ss_prop; ss_prop << m_Controls.m_proportion->value(); 
	std::stringstream ss_sigma; ss_sigma << m_Controls.spinGamma->value(); 
	std::stringstream ss_iter; ss_iter << m_Controls.mu_iter->value(); 
	std::stringstream ss_degree; ss_degree << m_Controls.m_degreeRemoval->value(); 

	std::ofstream _file_reconstruction;
	std::string fileName = writeFileBaseName + "_" + ss_prop.str() + "_" + ss_sigma.str() + "_" + ss_iter.str() + "_" + ss_degree.str() + "_RE.txt";

	if (std::remove(fileName.c_str()) != 0) std::cout << "No such file! " << std::endl;
	else std::cout << "File " << fileName << " has been removed !" << std::endl;

	_file_reconstruction.open(fileName.c_str(), ios::app);

	if (!_file_reconstruction)
	{
		std::cout << "file cannot open! " << std::endl;
		return;
	}

	_file_reconstruction << this->m_modelName << " model - dist of reconstruction with input : " << std::endl;

	double mean_RE = reconstructionErrors.mean();
	double SD = 0.0;
	for (int i = 0; i < m_dataSetsBackup.size(); i++)
	{
		_file_reconstruction << "GT " << i << " : " << std::setprecision(6) << reconstructionErrors[i] << std::endl;
		SD += pow((reconstructionErrors[i] - mean_RE), 2);
	}
		
	SD /= reconstructionErrors.size();
	SD = sqrt(SD);

	_file_reconstruction << "Average reconstruction error = " << std::setprecision(6) << mean_RE  << std::endl;
	_file_reconstruction << "SD = " << std::setprecision(6) << SD  << std::endl;
	_file_reconstruction.close();

	std::cout << std::endl; 
	std::cout << __FUNCTION__ << " : Distance of reconstructed output with input polys is " << std::setprecision(6) << mean_RE << " ( " << SD << " ) " << endl; 
	std::cout << std::endl; 
	std::cout << " ================================================ " << std::endl;

}


void NonlinearModelling::LoadTestGTForBackProjection()
{
	if (this->m_testGTForBP.size() > 0)
	{
		while (!m_testGTForBP.empty())
		{
			m_testGTForBP.pop_back();
		}
	}


	QFileDialog fd;
	fd.setDirectory(m_recentDir.c_str());
	fd.setFileMode(QFileDialog::Directory);
	QString selected;

	if (fd.exec() == QDialog::Accepted)
	{
		m_recentDir = fd.directory().absolutePath().toAscii();
		QStringList myfiles = fd.selectedFiles();
		if (!myfiles.isEmpty())
			selected = myfiles[0];
	}

	QDir* meshDir = new QDir(selected);
	QStringList::Iterator it;
	QStringList files = meshDir->entryList();
	it = files.begin();
	
	while (it != files.end())
	{
		if (QFileInfo(*meshDir, *it).isFile() && ((*it).endsWith)(".vtk"))
		{
			vtkSmartPointer< vtkPolyDataReader > pReader = vtkSmartPointer< vtkPolyDataReader >::New();
			pReader->SetFileName(QFileInfo(*meshDir, *it).absoluteFilePath().toAscii());
			pReader->Update();

			vtkSmartPointer< vtkPolyData > loadedPoly = pReader->GetOutput(); // loaded poly to be projected
			m_testGTForBP.push_back(loadedPoly);
		}
		++it;
	}

	VisualizeDatasets(this->m_testGTForBP, this->m_modelName + "_in", false);

	std::cout << endl;
	std::cout << __FUNCTION__ << " : Reloaded GT of Test Datasets for Back Projection " << m_testGTForBP.size() << " meshes ... " << std::endl;

}


void NonlinearModelling::LoadTestForBackProjection()
{
	std::cout << std::endl; 
	std::cout << "====================================================================" << std::endl; 
	std::cout << std::endl; 

	if (!m_robustKPCA)
	{
		MITK_WARN << " Please train a model ahead ! ";
		return;
	}

	if (this->m_testGTForBP.size() == 0)
	{
		MITK_WARN << " Please load ground truth for test datasets ! "; 
		return; 
	}

	std::string writeFileBaseName = m_Controls.printEvaluation->toPlainText().toStdString() + "_" + this->m_modelName; 


	// **************************************************************************************

	std::vector< vtkSmartPointer< vtkPolyData > > datasetsProjection;
	
	QFileDialog fd;
	fd.setDirectory(m_recentDir.c_str());
	fd.setFileMode(QFileDialog::Directory);
	QString selected;
	if (fd.exec() == QDialog::Accepted)
	{
		m_recentDir = fd.directory().absolutePath().toAscii();
		QStringList myfiles = fd.selectedFiles();
		if (!myfiles.isEmpty())
			selected = myfiles[0];
	}
	QDir* meshDir = new QDir(selected);
	QStringList::Iterator it;
	QStringList files = meshDir->entryList();
	it = files.begin();
	
	while (it != files.end())
	{
		if (QFileInfo(*meshDir, *it).isFile() && ((*it).endsWith)(".vtk"))
		{
			vtkSmartPointer< vtkPolyDataReader > pReader = vtkSmartPointer< vtkPolyDataReader >::New();
			pReader->SetFileName(QFileInfo(*meshDir, *it).absoluteFilePath().toAscii());
			pReader->Update();

			vtkSmartPointer< vtkPolyData > loadedPoly = pReader->GetOutput(); // loaded poly to be projected
			datasetsProjection.push_back(loadedPoly);
		}
		++it;
	}

	std::cout << endl; 
	std::cout << __FUNCTION__ << " : Compare the back projected polys with input " << datasetsProjection.size() << " meshes ... " << std::endl;

	if (datasetsProjection.size() != this->m_testGTForBP.size())
	{
		MITK_WARN << " GT and test datasets size are not equal ! "; 
		return; 
	}

	/////////////////////////////////////////////////////////////////////////////////////////////////////

	if (tthis->m_modelName == "RPCA")
	{
		for (int i = 0; i < datasetsProjection.size(); i++)
		{
			datasetsProjection.at(i)->DeepCopy(this->m_robustKPCA->ProjectShapeLinearModel(m_Controls.eval_proportion->value(), datasetsProjection.at(i), -1));
			datasetsProjection.at(i)->Modified();
		}
	}
	else if (this->m_modelName == "KPCA" || this->m_modelName == "Miccai17" || this->m_modelName == "RKPCA")
	{
		for (int i = 0; i < datasetsProjection.size(); i++)
		{
			datasetsProjection.at(i)->DeepCopy(this->m_robustKPCA->ProjectShapeNonlinearModel(m_Controls.eval_proportion->value(), datasetsProjection.at(i), -1));
			datasetsProjection.at(i)->Modified();
		}
	}

	VisualizeDatasets(datasetsProjection, writeFileBaseName + "_BP", false);

///////////////////////////////////////////////////////////////////////////////////////////////////////////

	vnl_vector< double > distances(datasetsProjection.size(), 0.0);

	for (int n = 0; n < datasetsProjection.size(); n++)
	{
		vtkPolyData* GTPoly = this->m_testGTForBP.at(n);
		vtkPolyData* testPoly = datasetsProjection.at(n);

//		distances[n] = this->ComputeEucliDistPolys(inputPoly, outputPoly); 

		distances[n] = this->AlignedAndComputeEucliDistPolys(GTPoly, testPoly); 
	}

	/* save to file */
	std::stringstream ss_prop; ss_prop << m_Controls.eval_proportion->value();  
	std::stringstream ss_sigma; ss_sigma << m_Controls.spinGamma->value();


	std::ofstream _file_backProjection;
	std::string fileName = writeFileBaseName + "_" + ss_prop.str() + "_" + ss_sigma.str() + "_BP.txt";

	if (std::remove(fileName.c_str()) != 0) std::cout << " No such file! " << std::endl;
	else std::cout << " File " << fileName << " has been removed !" << std::endl;

	_file_backProjection.open(fileName.c_str(), ios::app);

	if (!_file_backProjection)
	{
		std::cout << " file cannot open! " << std::endl;
		return;
	}

	_file_backProjection << this->m_modelName << " Back Projection Error :" << std::endl;
	 
	double meanDist = distances.mean();
	double SD = 0.0;
	for (int i = 0; i < distances.size(); i++)
	{
		_file_backProjection << "Test data " << i << " : " << std::setprecision(6) << distances[i] << std::endl;
		SD += pow((distances[i] - meanDist), 2);
	}

	SD /= distances.size();
	SD = sqrt(SD);

	_file_backProjection << "Average back projection error = " << std::setprecision(6) << meanDist << std::endl;
	_file_backProjection << "SD = " << std::setprecision(6) << SD << std::endl;
	_file_backProjection.close();

	std::cout << __FUNCTION__ << " : Average RE of " << datasetsProjection.size() << std::setprecision(6) << " datasets is " << meanDist << " ( " << SD << " ) . " << endl;
	std::cout << std::endl;
	std::cout << "==================================================================== " << std::endl;
}


//============================================ Utilities ===========================================================


void NonlinearModelling::SigmaEstimation()
{
	cout << __FUNCTION__ << " Start : " ; 

	std::vector< vtkSmartPointer<vtkPolyData> > estimatePolys; 

	QFileDialog fd;
	fd.setDirectory(m_recentDir.c_str());
	fd.setFileMode(QFileDialog::Directory);
	QString selected;

	if (fd.exec() == QDialog::Accepted)
	{
		m_recentDir = fd.directory().absolutePath().toAscii();

		QStringList myfiles = fd.selectedFiles();
		if (!myfiles.isEmpty())
			selected = myfiles[0];
	}

	QDir* meshDir = new QDir(selected);

	QStringList::Iterator it;
	QStringList files = meshDir->entryList();
	it = files.begin();

	while (it != files.end())
	{
		if (QFileInfo(*meshDir, *it).isFile() && ((*it).endsWith)(".vtk"))
		{
			vtkSmartPointer< vtkPolyDataReader > pReader = vtkSmartPointer< vtkPolyDataReader >::New();

			pReader->SetFileName(QFileInfo(*meshDir, *it).absoluteFilePath().toAscii());
			pReader->Update();

			vtkSmartPointer< vtkPolyData > loadedPoly = pReader->GetOutput();
			estimatePolys.push_back(loadedPoly); 
		}
		++it;
	}

	delete meshDir;


	double sigma = 0.0; 
	this->estimateKernelWidth(estimatePolys, sigma); 

	std::cout << __FUNCTION__ << " End ... " << endl;

}


void NonlinearModelling::estimateKernelWidth(std::vector< vtkSmartPointer< vtkPolyData >> datasets, double & EstimatedSigma)
{
	cout << __FUNCTION__ << " Start : with loaded " << datasets.size() << " datasets ... " << endl;

	if (datasets.size() == 0)
	{
		MITK_WARN << " Please load datasets first ! "; 
		return; 
	}

	std::vector< vtkSmartPointer< vtkPolyData > > dataBackUps; 
	for (int i = 0; i < datasets.size(); i++)
	{
		vtkPolyData* currPoly = datasets.at(i); 
		dataBackUps.push_back(currPoly); 
	}

	if (m_Controls.m_alignment->isChecked() == true)
	{
		this->CenterOfMassToOrigin(dataBackUps, false);

		this->AlignDataSetsWithoutScaling(dataBackUps);
	}
	
	std::vector< pdm::VectorOfDouble > DataVectors; 
	for (int i = 0; i < dataBackUps.size(); i++)
	{
		pdm::VectorOfDouble vec_i(dataBackUps.at(i)->GetNumberOfPoints() * 3, 0.0);
		for (int j = 0; j < dataBackUps.at(i)->GetNumberOfPoints(); j++)
		{
			vec_i[j * 3 + 0] = dataBackUps.at(i)->GetPoint(j)[0];
			vec_i[j * 3 + 1] = dataBackUps.at(i)->GetPoint(j)[1];
			vec_i[j * 3 + 2] = dataBackUps.at(i)->GetPoint(j)[2];
		}
		DataVectors.push_back(vec_i); 
	}

	std::vector< double > dist;
	for (int i = 0; i < DataVectors.size(); i++)
	{
		double dist_i_min = std::numeric_limits<double>::max();

		for (int j = 0; j < DataVectors.size(); j++)
		{	
			double dist_i_j = sqrt(vnl_vector_ssd(DataVectors.at(i), DataVectors.at(j)));

			dist.push_back(dist_i_j);
		}
	}

	double mean_dist = 0.0;
	double sigma_dist = 0.0;
	for (int i = 0; i < dist.size(); i++)
	{
		mean_dist += dist.at(i);
	}
	mean_dist /= dist.size();     

	EstimatedSigma = 0.5 * mean_dist;        

	std::cout << __FUNCTION__ << " mean_dist = " << mean_dist << " with size = " << dist.size() << std::endl;

	m_Controls.spinGamma->setValue(EstimatedSigma);

	cout << __FUNCTION__ << " End : with estimated Sigma = " << EstimatedSigma << endl;

}

void NonlinearModelling::AlignDataSetsWithScaling(std::vector< vtkSmartPointer< vtkPolyData >> &m_meshes)
{
	std::cout << __FUNCTION__ << " Start ..." ;
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
	std::cout << " meanScaling = " << meanScaling ;

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

	std::cout << " ... done. " << std::endl;
}


void NonlinearModelling::AlignDataSetsWithoutScaling(std::vector< vtkSmartPointer< vtkPolyData >> &m_meshes)
{
//	std::cout << __FUNCTION__ << " Start ... ";

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

//	std::cout << " done. " << std::endl;
}


void NonlinearModelling::CenterOfMassToOrigin(std::vector< vtkSmartPointer< vtkPolyData >> &m_meshes, bool weighting)
{
	std::cout << __FUNCTION__ << " Center meshes to Origin .. " ;
	for (unsigned int ns = 0; ns < m_meshes.size(); ns++)
	{
		vtkSmartPointer< vtkCenterOfMass > centerOfMassFilter = vtkSmartPointer< vtkCenterOfMass >::New();

		double center[3];

		if (weighting == false)
		{
			centerOfMassFilter->SetInputData(m_meshes.at(ns));
			centerOfMassFilter->SetUseScalarsAsWeights(false);
			centerOfMassFilter->Update();
			centerOfMassFilter->GetCenter(center);
		}
		else
		{
			centerOfMassFilter->ComputeCenterOfMass(m_meshes.at(ns)->GetPoints(), m_meshes.at(ns)->GetPointData()->GetArray("Weighting"), center);
		}

		// translate
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


void NonlinearModelling::ComputeGeneralizationLoO(std::vector< vtkSmartPointer< vtkPolyData >> trainPolys, vtkSmartPointer< vtkPolyData > toProjectPoly, vtkSmartPointer< vtkPolyData > projectGT, RobustKPCA::VectorOfDouble & _generalization)
{
	
	// create a model according to the model name with trainPolys 
	RobustKPCA* my_model = new RobustKPCA(m_Controls.spinGamma->value(), m_Controls.m_alignment->isChecked(), false);   // not scaling 

	my_model->ReadDataMatrix(trainPolys);

	
	/********************************************************************************************************/
	
	
	if (this->m_modelName == "RPCA")
	{
		my_model->RobustPCA();

		int preservedModes = std::min(_generalization.size(), my_model->GetParameterCount());

		for (int modeCnt = 1; modeCnt <= preservedModes; modeCnt++)
		{
			vtkPolyData* projectedPoly = vtkPolyData::New();
			projectedPoly->DeepCopy(my_model->ProjectShapeLinearModel(0, toProjectPoly, modeCnt));

			_generalization[modeCnt - 1] = this->AlignedAndComputeEucliDistPolys(projectGT, projectedPoly);
		}
	}

	if (this->m_modelName == "KPCA")
	{
		my_model->performKPCA();

		int preservedModes = std::min(_generalization.size(), my_model->GetParameterCount());

		for (int modeCnt = 1; modeCnt <= preservedModes; modeCnt++)
		{
			vtkPolyData* projectedPoly = vtkPolyData::New();
			projectedPoly->DeepCopy(my_model->ProjectShapeNonlinearModel(0, toProjectPoly, modeCnt));

			_generalization[modeCnt - 1] = this->AlignedAndComputeEucliDistPolys(projectGT, projectedPoly);
		}
	}

	if (this->m_modelName == "Miccai17")
	{
		my_model->Miccai17();

		int preservedModes = std::min(_generalization.size(), my_model->GetParameterCount());

		for (int modeCnt = 1; modeCnt <= preservedModes; modeCnt++)
		{
			vtkPolyData* projectedPoly = vtkPolyData::New();
			projectedPoly->DeepCopy(my_model->ProjectShapeNonlinearModel(0, toProjectPoly, modeCnt));

			_generalization[modeCnt - 1] = this->AlignedAndComputeEucliDistPolys(projectGT, projectedPoly);
		}
	}

	if (this->m_modelName == "RKPCA")
	{
		my_model->RKPCA(m_Controls.mu_iter->value(), 0);

		int preservedModes = std::min(_generalization.size(), my_model->GetParameterCount());

		for (int modeCnt = 1; modeCnt <= preservedModes; modeCnt++)
		{
			vtkPolyData* projectedPoly = vtkPolyData::New();
			projectedPoly->DeepCopy(my_model->ProjectShapeNonlinearModel(0, toProjectPoly, modeCnt));

			_generalization[modeCnt - 1] = this->AlignedAndComputeEucliDistPolys(projectGT, projectedPoly);
		}
	}

	delete my_model; 


}


double NonlinearModelling::AlignedAndComputeEucliDistPolys(vtkSmartPointer< vtkPolyData > polyGT, vtkSmartPointer< vtkPolyData > polyTest)
{
	std::vector< vtkSmartPointer< vtkPolyData >> tmpPolys; 
	tmpPolys.push_back(polyGT);
	tmpPolys.push_back(polyTest); 

	this->AlignDataSetsWithoutScaling(tmpPolys);

//	this->CenterOfMassToOrigin(tmpPolys, false); 

	this->ICPAlignment(tmpPolys.at(0), tmpPolys.at(1));

	/*this->AddPolyToNode(tmpPolys.at(0), "tmp0", false);
	this->AddPolyToNode(tmpPolys.at(1), "tmp1", false);*/
	
	return this->ComputeEucliDistPolys(tmpPolys.at(0), tmpPolys.at(1)); 
}


double NonlinearModelling::ComputeEucliDistPolys(vtkSmartPointer< vtkPolyData > poly1, vtkSmartPointer< vtkPolyData > poly2)
{
	if (poly1->GetNumberOfPoints() != poly2->GetNumberOfPoints())
	{
		MITK_WARN << __FUNCTION__ << " poly dimensions do not match " << endl; 
		return -1; 
	}

	double dist = 0.0; 
	for (int i = 0; i < poly1->GetNumberOfPoints() ; i ++ )
	{
		double* pos1 = poly1->GetPoint(i); 
		double* pos2 = poly2->GetPoint(i); 

		double temp = pow(pos1[0] - pos2[0], 2) + pow(pos1[1] - pos2[1], 2) + pow(pos1[2] - pos2[2], 2);

		dist += sqrt(temp);
	}

	return dist / poly1->GetNumberOfPoints(); 

}


void NonlinearModelling::ICPAlignment(vtkSmartPointer< vtkPolyData > _polyGT, vtkSmartPointer< vtkPolyData > & _polyToAlign)
{
	vtkSmartPointer< vtkIterativeClosestPointTransform > icp = vtkSmartPointer< vtkIterativeClosestPointTransform >::New();
	icp->SetSource(_polyToAlign);
	icp->SetTarget(_polyGT);
	//	icp->GetLandmarkTransform()->SetModeToRigidBody();
	icp->GetLandmarkTransform()->SetModeToSimilarity();  // size ?? 
	icp->GetLandmarkTransform()->Modified();
	icp->StartByMatchingCentroidsOn();
	icp->SetMaximumNumberOfIterations(50);
	icp->Modified();
	icp->Update();

	vtkSmartPointer<vtkTransformPolyDataFilter> icpTransformFilter = vtkSmartPointer<vtkTransformPolyDataFilter>::New();
	icpTransformFilter->SetInputData(_polyToAlign);
	icpTransformFilter->SetTransform(icp);
	icpTransformFilter->Update();

	_polyToAlign->DeepCopy(icpTransformFilter->GetOutput());
	_polyToAlign->Modified();

}


void NonlinearModelling::ComputeNormals(vtkPolyData* _poly)
{
	vtkSmartPointer<vtkPolyDataNormals> normalsGen = vtkSmartPointer<vtkPolyDataNormals>::New();

	normalsGen->SplittingOff();
	normalsGen->ConsistencyOff();
	//	normalsGen->AutoOrientNormalsOff();
	normalsGen->AutoOrientNormalsOn();
	normalsGen->ComputePointNormalsOn();
	normalsGen->SetFeatureAngle(180);

	//	normalsGen->SplittingOn();  // preserve sharp edges
	//	normalsGen->ConsistencyOn(); // polygon order 
	////	normalsGen->ConsistencyOff();
	//	normalsGen->AutoOrientNormalsOn(); 
	////	normalsGen->AutoOrientNormalsOff();
	//	normalsGen->ComputePointNormalsOn();
	//	normalsGen->ComputeCellNormalsOff(); 
	//	normalsGen->SetFeatureAngle(180);
	////	normalsGen->SetSplitting(0); 

	normalsGen->SetInputData(_poly);
	normalsGen->Update();

	_poly->DeepCopy(normalsGen->GetOutput());
	_poly->Modified();
}


vtkSmartPointer< vtkIdList > NonlinearModelling::GetConnectedVertices(vtkSmartPointer< vtkPolyData > _poly, int _id)
{
	vtkSmartPointer< vtkIdList > connectedVertices = vtkSmartPointer< vtkIdList >::New();

	// get all cells that vertex 'id' is a part of 
	vtkSmartPointer< vtkIdList > cellIdList = vtkSmartPointer< vtkIdList >::New();
	_poly->GetPointCells(_id, cellIdList);

	for (vtkIdType i = 0; i < cellIdList->GetNumberOfIds(); i++)
	{
		vtkSmartPointer< vtkIdList > pointIdList = vtkSmartPointer< vtkIdList >::New();
		_poly->GetCellPoints(cellIdList->GetId(i), pointIdList);

		if (pointIdList->GetId(0) != _id)
		{
			connectedVertices->InsertNextId(pointIdList->GetId(0));
		}
		else
		{
			connectedVertices->InsertNextId(pointIdList->GetId(1));
		}
	}

	return connectedVertices;

}


void NonlinearModelling::InsertIdsIntoArray(std::vector< int >  & _ids, vtkSmartPointer< vtkIdList > _connectedVertices)
{
	for (vtkIdType i = 0; i < _connectedVertices->GetNumberOfIds(); i++)
	{
		bool insert = true;
		for (int cnt = 0; cnt < _ids.size(); cnt++)
		{
			if (_ids.at(cnt) == _connectedVertices->GetId(i))
			{
				insert = false;
				break;
			}
		}

		if (insert == true) _ids.push_back(_connectedVertices->GetId(i));
	}

}
