#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "sphere.h"
#include "plane.h"
#include "camera.h"
#include "Renderer.h"
#include "LightSource.h"
#include <stdlib.h>
#include "cone.h"
#include "color.h"

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

	connect(ui->addPrimitivesBox, SIGNAL(currentIndexChanged(int)), this, SLOT(addToSelectionPrimitives(int)));

	//Working woth Lights

	connect(ui->light_x, &QDoubleSpinBox::textChanged, this, &MainWindow::onLightPositionChangeButtonClicked);
	connect(ui->light_y, &QDoubleSpinBox::textChanged, this, &MainWindow::onLightPositionChangeButtonClicked);
	connect(ui->light_z, &QDoubleSpinBox::textChanged, this, &MainWindow::onLightPositionChangeButtonClicked);

	connect(ui->light_r, &QDoubleSpinBox::textChanged, this, &MainWindow::onLightColorChangeButtonClicked);
	connect(ui->light_g, &QDoubleSpinBox::textChanged, this, &MainWindow::onLightColorChangeButtonClicked);
	connect(ui->light_b, &QDoubleSpinBox::textChanged, this, &MainWindow::onLightColorChangeButtonClicked);





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
void MainWindow::onLightColorChangeButtonClicked()
{
	ColorRGB newColor = ColorRGB(ui->light_r->value(), ui->light_g->value(), ui->light_b->value());
	ui->graphicsView->_sceneManager->getScene()->getLightSource()->setColor(newColor);
}

void MainWindow::addToSelectionPrimitives(int idx)
{
	if (idx == ui->graphicsView->add_sphere_idx)
	{
		int spheres_count = ui->graphicsView->spheres_count;
		ui->choose_primitives_box->addItem(("Сфера" + std::to_string(spheres_count)).c_str());
	}
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
