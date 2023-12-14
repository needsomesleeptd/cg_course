#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <QFileDialog>
#include <QMessageBox>
#include <QKeyEvent>
#include <memory>

#include "sceneManager.h"
#include "sceneManagerCreator.h"

QT_BEGIN_NAMESPACE
namespace Ui
{
	class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
 Q_OBJECT

 public:
	MainWindow(QWidget* parent = nullptr);
	~MainWindow();

 public slots:
		void addToSelectionPrimitives(int idx);
		void onTranslateButtonClicked();
		void onRotateButtonClicked();
		void onAddButtonClicked();
		void currentShapeChanged(int shape_idx);
		void MainWindowFPSDisplay();
		void onMeasureTimeClicked();

 protected:
	//void resizeEvent(QResizeEvent *event) override;
	void setupScene();
	void updateScene();


 private:
	Ui::MainWindow* ui;





	void onLightPositionChangeButtonClicked();
	void onLightColorChangeButtonClicked();

	void materialUpdate();







};

#endif // MAINWINDOW_H
