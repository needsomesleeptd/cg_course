#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "sphere.h"
#include "plane.h"
#include "camera.h"
#include "Renderer.h"
#include "LightSource.h"
#include <stdlib.h>
#include "cone.h"

float GenerateRandom()
{

	return rand() / (RAND_MAX + 1.);
}

MainWindow::MainWindow(QWidget* parent)
	: QMainWindow(parent), ui(new Ui::MainWindow)
{
	ui->setupUi(this);

	ui->addPrimitivesBox->addItem("Добавить сферу");
	ui->addPrimitivesBox->addItem("Добавить конус");
	ui->addPrimitivesBox->addItem("Добавить куб");
	ui->addPrimitivesBox->addItem("Добавить циллиндр");

	connect(ui->addPrimitivesBox, SIGNAL(currentIndexChanged(int)), ui->graphicsView, SLOT(addPrimitive(int)));

	connect(ui->light_x, &QDoubleSpinBox::textChanged, this, &MainWindow::onLightPositionChangeButtonClicked);
	connect(ui->light_y, &QDoubleSpinBox::textChanged, this, &MainWindow::onLightPositionChangeButtonClicked);
	connect(ui->light_z, &QDoubleSpinBox::textChanged, this, &MainWindow::onLightPositionChangeButtonClicked);





	/*connect(ui->pushButton_del_model_cur, &QPushButton::clicked, this, &MainWindow::onRemoveModelButtonClicked);
	connect(ui->pushButton_del_model_all, &QPushButton::clicked, this, &MainWindow::onRemoveAllModelsButtonClicked);

	connect(ui->pushButton_add_camera, &QPushButton::clicked, this, &MainWindow::onAddCameraButtonClicked);
	connect(ui->pushButton_del_camera_cur, &QPushButton::clicked, this, &MainWindow::onRemoveCameraButtonClicked);*/



	/*connect(ui->pushButton_move, &QPushButton::clicked, this, &MainWindow::onMoveButtonClicked);
	connect(ui->pushButton_move_all, &QPushButton::clicked, this, &MainWindow::onMoveAllButtonClicked);

	connect(ui->pushButton_scale, &QPushButton::clicked, this, &MainWindow::onScaleButtonClicked);
	connect(ui->pushButton_scale_all, &QPushButton::clicked, this, &MainWindow::onScaleAllButtonClicked);

	connect(ui->pushButton_spin, &QPushButton::clicked, this, &MainWindow::onRotateButtonClicked);
	connect(ui->pushButton_spin_all, &QPushButton::clicked, this, &MainWindow::onRotateAllButtonClicked);

	connect(ui->comboBox_cameras, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::changeCam);*/
}
void MainWindow::onCameraRotateBbuttonClicked()
{

}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::setupScene()
{

}
void MainWindow::updateScene()
{

}
void MainWindow::onLightPositionChangeButtonClicked()
{
	ui->graphicsView->_sceneManager->getScene()->getLightSource()->setPosition(VecD3(ui->light_x->value(),
		ui->light_y->value(),
		ui->light_z->value()));
}
