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
	ui->addPrimitivesBox->addItem("Добавить цилиндр");

	connect(ui->translate, SIGNAL(clicked()), this, SLOT(onTranslateButtonClicked()));

	connect(ui->add, SIGNAL(clicked()), this, SLOT(onAddButtonClicked()));



	//Working woth Lights

	connect(ui->light_x, &QDoubleSpinBox::textChanged, this, &MainWindow::onLightPositionChangeButtonClicked);
	connect(ui->light_y, &QDoubleSpinBox::textChanged, this, &MainWindow::onLightPositionChangeButtonClicked);
	connect(ui->light_z, &QDoubleSpinBox::textChanged, this, &MainWindow::onLightPositionChangeButtonClicked);

	connect(ui->light_r, &QDoubleSpinBox::textChanged, this, &MainWindow::onLightColorChangeButtonClicked);
	connect(ui->light_g, &QDoubleSpinBox::textChanged, this, &MainWindow::onLightColorChangeButtonClicked);
	connect(ui->light_b, &QDoubleSpinBox::textChanged, this, &MainWindow::onLightColorChangeButtonClicked);

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
	if (idx == ui->graphicsView->add_cone_idx)
	{
		int cones_count = ui->graphicsView->cones_count;
		ui->choose_primitives_box->addItem(("Конус" + std::to_string(cones_count)).c_str());
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
void MainWindow::onTranslateButtonClicked()
{
	int idx = ui->choose_primitives_box->currentIndex();
	VecD3 moveParams = VecD3(ui->obj_x->value(), ui->obj_y->value(), ui->obj_z->value());
	qDebug() << "started moving on" << idx << moveParams.x << moveParams.y << moveParams.z;
	ui->graphicsView->movePrimitive(idx, moveParams);
}
void MainWindow::onAddButtonClicked()
{
	int idx = ui->addPrimitivesBox->currentIndex();
	addToSelectionPrimitives(idx);
	qDebug() << "adding primitive with id" << idx;
	ui->graphicsView->addPrimitive(idx);
}
