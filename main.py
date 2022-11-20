from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from ClientClips import MakeHitAreaFromSequence
from design import Ui_MainWindow
from queue import Queue
import sys


class WriteStream(object):
    def __init__(self,queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)
    
    def flush(self):
        pass

class MyReceiver(QObject):
    mysignal = pyqtSignal(str)

    def __init__(self,queue,*args,**kwargs):
        QObject.__init__(self,*args,**kwargs)
        self.queue = queue

    @pyqtSlot()
    def run(self):
        while True:
            text = self.queue.get()
            self.mysignal.emit(text)


class MainWindow(QMainWindow):
    
    def __init__(self):
        
        QMainWindow.__init__(self)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.Ui_components()

    def Ui_components(self):
        self.ui.pushButton.clicked.connect(self.get_input_path)
        self.ui.pushButton_2.clicked.connect(self.get_output_path)
        self.ui.pushButton_3.clicked.connect(self.run)
        self.ui.textEdit.setReadOnly(True)

    @pyqtSlot(str)
    def append_text(self,text):
        self.ui.textEdit.moveCursor(QTextCursor.End)
        self.ui.textEdit.insertPlainText( text )

    def run(self):
        input_path = self.ui.lineEdit.text().strip()
        output_path = self.ui.lineEdit_2.text().strip()
        if len(input_path) == 0 or len(output_path) == 0:
            self.append_text("Input Path and Output Path cannot be empty!\n")
            return
        runner = MakeHitAreaFromSequence(input_path, output_path)
        runner.main()

    def get_input_path(self, *args):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if folder:
            self.ui.lineEdit.setText(folder)
        
    def get_output_path(self, *args):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if folder:
            self.ui.lineEdit_2.setText(folder)


def main():
    # Create Queue and redirect sys.stdout to this queue
    queue = Queue()
    sys.stdout = WriteStream(queue)

    # Create QApplication and QWidget
    qapp = QApplication(sys.argv)  
    app = MainWindow()
    app.show()

    # Create thread that will listen on the other end of the queue, and send the text to the textedit in our application
    thread = QThread()
    my_receiver = MyReceiver(queue)
    my_receiver.mysignal.connect(app.append_text)
    my_receiver.moveToThread(thread)
    thread.started.connect(my_receiver.run)
    thread.start()
    # Start the mainloop for the GUI/window
    qapp.exec_()


if __name__ == "__main__":
    main()